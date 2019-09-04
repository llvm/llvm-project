#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <inttypes.h>

#include "cilksan_internal.h"
#include "debug_util.h"
#include "disjointset.h"
#include "frame_data.h"
#include "race_detect_update.h"
#include "shadow_mem.h"
#include "spbag.h"
#include "stack.h"

#if CILKSAN_DEBUG
enum EventType_t last_event = NONE;
static bool CILKSAN_INITIALIZED = false;
#endif

// declared in driver.cpp
extern FILE *err_io;
// declared in print_addr.cpp
extern uintptr_t *load_pc;
extern uintptr_t *store_pc;

// --------------------- stuff from racedetector ---------------------------

// -------------------------------------------------------------------------
//  Analysis data structures and fields
// -------------------------------------------------------------------------

// List used for the disjoint set data structure's find_set operation.
List_t disjoint_set_list;

#if CILKSAN_DEBUG
template<>
long DisjointSet_t<SPBagInterface *>::debug_count = 0;

long SBag_t::debug_count = 0;
long PBag_t::debug_count = 0;
#endif

void free_bag(DisjointSet_t<SPBagInterface *> *ptr) {
  delete ptr->get_node();
}

template<>
void (*DisjointSet_t<SPBagInterface *>::dtor_callback)(DisjointSet_t *) =
  &free_bag;

// // Free list for disjoint-set nodes
// template<>
// DisjointSet_t<SPBagInterface *> *
// DisjointSet_t<SPBagInterface *>::free_list = nullptr;

// Free lists for SBags and PBags
SBag_t::FreeNode_t *SBag_t::free_list = nullptr;
PBag_t::FreeNode_t *PBag_t::free_list = nullptr;

// Code to handle references to the stack.

// Range of stack used by the process
uintptr_t stack_low_addr = (uintptr_t)-1;
uintptr_t stack_high_addr = 0;

// Helper function to check if an address is in the stack.
static inline bool is_on_stack(uintptr_t addr) {
  // cilksan_assert(stack_high_addr != stack_low_addr);
  return (addr <= stack_high_addr && addr >= stack_low_addr);
}

// Free list for call-stack nodes
call_stack_node_t *call_stack_node_t::free_list = nullptr;

// Global object to manage Cilksan data structures.
CilkSanImpl_t CilkSanImpl;

template<>
DisjointSet_t<SPBagInterface *>::DJSAllocator
&DisjointSet_t<SPBagInterface *>::Alloc = CilkSanImpl.getDJSAllocator();

////////////////////////////////////////////////////////////////////////
// Events functions
////////////////////////////////////////////////////////////////////////

/// Helper function for merging returning child's bag into parent's
inline void
CilkSanImpl_t::merge_bag_from_returning_child(bool returning_from_detach) {
  FrameData_t *parent = frame_stack.ancestor(1);
  FrameData_t *child = frame_stack.head();
  cilksan_assert(parent->Sbag);
  cilksan_assert(child->Sbag);
  // cilksan_assert(!child->Pbag);

  if (returning_from_detach) {
    // We are returning from a detach.  Merge the child S- and P-bags
    // into the parent P-bag.

    // Get the parent P-bag.
    DisjointSet_t<SPBagInterface *> *parent_pbag = parent->Pbag;
    if (!parent_pbag) { // lazily create PBag when needed
      DBG_TRACE(DEBUG_BAGS,
                "frame %ld creates a PBag ",
                parent->Sbag->get_set_node()->get_func_id());
      parent_pbag =
        new DisjointSet_t<SPBagInterface *>(new PBag_t(parent->Sbag->get_node()));
      parent->set_pbag(parent_pbag);
      DBG_TRACE(DEBUG_BAGS, "%p\n", parent_pbag);
    }

    cilksan_assert(parent_pbag && parent_pbag->get_set_node()->is_PBag());
    // Combine child S-bag into parent P-bag.
    cilksan_assert(child->Sbag->get_set_node()->is_SBag());
    if (child->is_Sbag_used()) {
      DBG_TRACE(
          DEBUG_BAGS,
          "Merge S-bag from detached child %ld to P-bag from parent %ld.\n",
          child->Sbag->get_set_node()->get_func_id(),
          parent->Sbag->get_set_node()->get_func_id());
      parent_pbag->combine(child->Sbag);
      parent->set_Pbag_used();
      // The child entry of frame_stack keeps child->Sbag alive until its bags
      // are reset at the end of this function.
      cilksan_assert(child->Sbag->get_set_node()->is_PBag());
    }

    // Combine child P-bag into parent P-bag.
    if (child->Pbag) {
      cilksan_assert(child->Pbag->get_set_node()->is_PBag());
      // if (child->is_Pbag_used()) {
      DBG_TRACE(DEBUG_BAGS,
                "Merge P-bag from spawned child %ld to P-bag from parent %ld.\n",
                child->Sbag->get_set_node()->get_func_id(),
                parent->Sbag->get_set_node()->get_func_id());
      parent_pbag->combine(child->Pbag);
      parent->set_Pbag_used();
      cilksan_assert(child->Pbag->get_set_node()->is_PBag());
      // }
    }

  } else {
    // We are returning from a call.  Merge the child S-bag into the
    // parent S-bag, and merge the child P-bag into the parent P-bag.
    cilksan_assert(parent->Sbag->get_set_node()->is_SBag());
    if (child->is_Sbag_used()) {
      DBG_TRACE(DEBUG_BAGS,
                "Merge S-bag from called child %ld to S-bag from parent %ld.\n",
                child->Sbag->get_set_node()->get_func_id(),
                parent->Sbag->get_set_node()->get_func_id());
      parent->Sbag->combine(child->Sbag);
      parent->set_Sbag_used();
    }

    // Combine child P-bag into parent P-bag.
    if (child->Pbag) {
      // Get the parent P-bag.
      DisjointSet_t<SPBagInterface *> *parent_pbag = parent->Pbag;
      if (!parent_pbag) { // lazily create PBag when needed
        DBG_TRACE(DEBUG_BAGS,
                  "frame %ld creates a PBag ",
                  parent->Sbag->get_set_node()->get_func_id());
        parent_pbag =
          new DisjointSet_t<SPBagInterface *>(new PBag_t(parent->Sbag->get_node()));
        parent->set_pbag(parent_pbag);
        DBG_TRACE(DEBUG_BAGS, "%p\n", parent_pbag);
      }

      cilksan_assert(parent_pbag && parent_pbag->get_set_node()->is_PBag());

      // Combine child P-bag into parent P-bag.
      cilksan_assert(child->Pbag->get_set_node()->is_PBag());
      // if (child->is_Pbag_used()) {
      DBG_TRACE(DEBUG_BAGS,
                "Merge P-bag from called child %ld to P-bag from parent %ld.\n",
                child->frame_data.frame_id,
                parent->Sbag->get_set_node()->get_func_id());
      parent_pbag->combine(child->Pbag);
      parent->set_Pbag_used();
      cilksan_assert(child->Pbag->get_set_node()->is_PBag());
      // }
    }
  }
  DBG_TRACE(DEBUG_BAGS, "After merge, parent set node func id: %ld.\n",
            parent->Sbag->get_set_node()->get_func_id());
  cilksan_assert(parent->Sbag->get_node()->get_func_id() ==
                 parent->Sbag->get_set_node()->get_func_id());

  // Reset child's bags.
  child->set_sbag(NULL);
  child->set_pbag(NULL);
}

/// Helper function for handling the start of a new function.  This
/// function can be a spawned or called Cilk function or a spawned C
/// function.  A called C function is treated as inlined.
inline void CilkSanImpl_t::start_new_function() {
  frame_id++;
  frame_stack.push();

  DBG_TRACE(DEBUG_CALLBACK, "Enter frame %ld, ", frame_id);

  // Get the parent pointer after we push, because once pused, the
  // pointer may no longer be valid due to resize.
  FrameData_t *parent = frame_stack.ancestor(1);
  DBG_TRACE(DEBUG_CALLBACK, "parent frame %ld.\n", parent->frame_data.frame_id);
  DisjointSet_t<SPBagInterface *> *child_sbag;

  FrameData_t *child = frame_stack.head();
  cilksan_assert(child->Sbag == NULL);
  cilksan_assert(child->Pbag == NULL);

  DBG_TRACE(DEBUG_BAGS, "Creating SBag for frame %ld\n", frame_id);
  child_sbag =
    new DisjointSet_t<SPBagInterface *>(new SBag_t(frame_id, call_stack));

  child->init_new_function(child_sbag);

  // We do the assertion after the init so that ref_count is 1.
  cilksan_assert(child_sbag->get_set_node()->is_SBag());

  WHEN_CILKSAN_DEBUG(frame_stack.head()->frame_data.frame_id = frame_id);

  DBG_TRACE(DEBUG_CALLBACK, "Enter function id %ld\n", frame_id);
}

/// Helper function for exiting a function; counterpart of start_new_function.
inline void CilkSanImpl_t::exit_function() {
  // Popping doesn't actually destruct the object so we need to
  // manually dec the ref counts here.
  frame_stack.head()->reset();
  frame_stack.pop();
}

/// Action performed on entering a Cilk function (excluding spawn helper).
inline void CilkSanImpl_t::enter_cilk_function() {
  DBG_TRACE(DEBUG_CALLBACK, "entering a Cilk function, push frame_stack\n");
  start_new_function();
}

/// Action performed on leaving a Cilk function (excluding spawn helper).
inline void CilkSanImpl_t::leave_cilk_function() {
  DBG_TRACE(DEBUG_CALLBACK,
            "leaving a Cilk function (spawner or helper), pop frame_stack\n");

  // param: not returning from a spawn
  merge_bag_from_returning_child(0);
  exit_function();
}

/// Action performed on entering a spawned child, right after detach.
inline void CilkSanImpl_t::enter_detach_child() {
  DBG_TRACE(DEBUG_CALLBACK, "done detach, push frame_stack\n");
  start_new_function();
  // Copy the rsp from the parent.
  FrameData_t *detached = frame_stack.head();
  FrameData_t *parent = frame_stack.ancestor(1);
  detached->Sbag->get_node()->set_rsp(parent->Sbag->get_node()->get_rsp());
  // Set the frame data.
  frame_stack.head()->frame_data.entry_type = DETACHER;
  frame_stack.head()->frame_data.frame_type = SHADOW_FRAME;
  DBG_TRACE(DEBUG_CALLBACK, "new detach frame started\n");
}

/// Action performed when returning from a spawned child.
/// (That is, returning from a spawn helper.)
inline void CilkSanImpl_t::return_from_detach() {
  DBG_TRACE(DEBUG_CALLBACK, "return from detach, pop frame_stack\n");
  cilksan_assert(DETACHER == frame_stack.head()->frame_data.entry_type);
  // param: we are returning from a spawn
  merge_bag_from_returning_child(1);
  exit_function();
  // // Detacher frames do not have separate leave calls from the helpers
  // // containing them, so we manually call leave_cilk_function again.
  // leave_cilk_function();
}

/// Action performed immediately after passing a sync.
inline void CilkSanImpl_t::complete_sync() {
  FrameData_t *f = frame_stack.head();
  DBG_TRACE(DEBUG_CALLBACK, "frame %d done sync\n",
            f->Sbag->get_node()->get_func_id());

  cilksan_assert(f->Sbag->get_set_node()->is_SBag());
  // Pbag could be NULL if we encounter a sync without any spawn (i.e., any Cilk
  // function that executes the base case)
  if (f->Pbag) {
    DBG_TRACE(DEBUG_BAGS,
              "Merge P-bag in frame %ld into S-bag.\n",
              f->Sbag->get_set_node()->get_func_id());
    cilksan_assert(f->Pbag->get_set_node()->is_PBag());
    // if (f->is_Pbag_used()) {
    f->Sbag->combine(f->Pbag);
    cilksan_assert(f->Pbag->get_set_node()->is_SBag());
    f->set_Sbag_used();
    // }
    cilksan_assert(f->Sbag->get_node()->get_func_id() ==
                   f->Sbag->get_set_node()->get_func_id());
    f->set_pbag(NULL);
  }
}

//---------------------------------------------------------------
// Callback functions
//---------------------------------------------------------------
void CilkSanImpl_t::do_enter_begin() {
  cilksan_assert(CILKSAN_INITIALIZED);
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = ENTER_FRAME);
  DBG_TRACE(DEBUG_CALLBACK,
            "frame %ld cilk_enter_frame_begin, stack depth %d\n",
            frame_id+1, frame_stack.size());
  // entry_stack.push();
  // entry_stack.head()->entry_type = SPAWNER;
  // entry_stack.head()->frame_type = SHADOW_FRAME;
  // entry_stack always gets pushed slightly before frame_id gets incremented
  // WHEN_CILKSAN_DEBUG(entry_stack.head()->frame_id = frame_id+1);
  enter_cilk_function();
  frame_stack.head()->frame_data.entry_type = SPAWNER;
  frame_stack.head()->frame_data.frame_type = SHADOW_FRAME;
}

void CilkSanImpl_t::do_enter_helper_begin() {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_CALLBACK, "frame %ld cilk_enter_helper_begin\n", frame_id+1);
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = ENTER_HELPER;);

  // entry_stack.push();
  // entry_stack.head()->entry_type = HELPER;
  // entry_stack.head()->frame_type = SHADOW_FRAME;
  // entry_stack always gets pushed slightly before frame_id gets incremented
  // WHEN_CILKSAN_DEBUG(entry_stack.head()->frame_id = frame_id+1;);
  // WHEN_CILKSAN_DEBUG(update_deque_for_entering_helper(););
  enter_cilk_function();
  // frame_stack.head()->frame_data.entry_type = HELPER;
  frame_stack.head()->frame_data.entry_type = DETACHER;
  frame_stack.head()->frame_data.frame_type = SHADOW_FRAME;
}

void CilkSanImpl_t::do_enter_end(uintptr_t stack_ptr) {
  cilksan_assert(CILKSAN_INITIALIZED);
  FrameData_t *cilk_func = frame_stack.head();
  cilk_func->Sbag->get_node()->set_rsp(stack_ptr);
  cilksan_assert(last_event == ENTER_FRAME || last_event == ENTER_HELPER);
  WHEN_CILKSAN_DEBUG(last_event = NONE);
  DBG_TRACE(DEBUG_CALLBACK, "cilk_enter_end, frame stack ptr: %p\n", stack_ptr);
}

void CilkSanImpl_t::do_detach_begin() {
  cilksan_assert(CILKSAN_INITIALIZED);
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = DETACH);
}

void CilkSanImpl_t::do_detach_end() {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_CALLBACK, "cilk_detach\n");

  cilksan_assert(last_event == DETACH);
  WHEN_CILKSAN_DEBUG(last_event = NONE);

  // At this point, the frame_stack.head is still the parent (spawning) frame
  WHEN_CILKSAN_DEBUG({
      FrameData_t *parent = frame_stack.head();
      DBG_TRACE(DEBUG_CALLBACK,
                "frame %ld about to spawn.\n",
                parent->Sbag->get_node()->get_func_id());
    });

  // if (!parent->Pbag) { // lazily create PBag when needed
  //   DBG_TRACE(DEBUG_BAGS,
  //          "frame %ld creates a PBag.\n",
  //          parent->Sbag->get_set_node()->get_func_id());
  //   DisjointSet_t<SPBagInterface *> *parent_pbag =
  //     new DisjointSet_t<SPBagInterface *>(new PBag_t(parent->Sbag->get_node()));
  //   parent->set_pbag(parent_pbag);
  // }
  // enter_detach_child();
}

void CilkSanImpl_t::do_loop_iteration_begin(uintptr_t stack_ptr) {
  DBG_TRACE(DEBUG_CALLBACK, "do_loop_iteration_begin()\n");
  if (start_new_loop) {
    // The first time we enter the loop, create a LOOP_FRAME at the head of
    // frame_stack.
    DBG_TRACE(DEBUG_CALLBACK, "starting new loop\n");
    // Start a new frame.
    do_enter_helper_begin();
    // Set this frame's type as LOOP_FRAME.
    FrameData_t *func = frame_stack.head();
    func->frame_data.frame_type = LOOP_FRAME;
    // Create a new iteration bag for this frame.
    DBG_TRACE(DEBUG_BAGS,
              "frame %ld creates an Iter-bag ",
              func->Sbag->get_set_node()->get_func_id());
    func->create_iterbag();
    DBG_TRACE(DEBUG_BAGS, "%p\n", func->Iterbag);
    // Finish initializing the frame.
    do_enter_end(stack_ptr);
    func->Iterbag->get_node()->set_rsp(stack_ptr);
    do_detach_begin();
    do_detach_end();
    start_new_loop = false;
  } else {
    FrameData_t *func = frame_stack.head();
    cilksan_assert(in_loop());
    func->Sbag->get_set_node()->set_rsp(stack_ptr);
    func->Iterbag->get_node()->set_rsp(stack_ptr);
  }
}

void CilkSanImpl_t::do_loop_iteration_end() {
  // frame_stack.head()->Sbag->set_version(
  //     frame_stack.head()->Sbag->get_node()->get_version());

  // At the end of each iteration, update the LOOP_FRAME for reuse.
  DBG_TRACE(DEBUG_CALLBACK, "do_loop_iteration_end()\n");
  FrameData_t *func = frame_stack.head();
  cilksan_assert(in_loop());
  // Get this frame's P-bag, creating it if necessary.
  DisjointSet_t<SPBagInterface *> *my_pbag = func->Pbag;
  if (!my_pbag) {
    DBG_TRACE(DEBUG_BAGS,
              "frame %ld creates a P-bag ",
              func->Sbag->get_set_node()->get_func_id());
    my_pbag =
      new DisjointSet_t<SPBagInterface *>(
          new PBag_t(func->Sbag->get_node()));
    func->set_pbag(my_pbag);
    DBG_TRACE(DEBUG_BAGS, "%p\n", my_pbag);
  }
  cilksan_assert(my_pbag && my_pbag->get_set_node()->is_PBag());

  // Merge the S-bag into the P-bag.
  //
  // TODO: Make the combine and recreation of the S-bag conditional on the S-bag
  // having references to it.
  DisjointSet_t<SPBagInterface *> *my_sbag = func->Sbag;
  uint64_t func_id = my_sbag->get_set_node()->get_func_id();
  if (func->is_Sbag_used()) {
    DBG_TRACE(DEBUG_BAGS,
              "Merge S-bag in loop frame %ld into P-bag.\n", func_id);
    my_pbag->combine(my_sbag);
    func->set_Pbag_used();
    cilksan_assert(func->Sbag->get_set_node()->is_PBag());

    // Create a new S-bag for the frame.
    DBG_TRACE(DEBUG_BAGS, "frame %ld creates an S-bag ", func_id);
    func->set_sbag(new DisjointSet_t<SPBagInterface *>(
                       new SBag_t(func_id, call_stack)));
    my_pbag->get_node()->update_sibling(func->Sbag->get_node());
    DBG_TRACE(DEBUG_BAGS, "%p\n", func->Sbag);
  }

  // Increment the Iter-frame.
  if (!func->inc_version()) {
    // Combine the Iter-bag into this P-bag.
    if (func->is_Iterbag_used()) {
      DBG_TRACE(DEBUG_BAGS,
                "Merge Iter-bag in loop frame %ld into P-bag.\n", func_id);
      my_pbag->combine(func->Iterbag);
      func->set_Pbag_used();
      cilksan_assert(func->Iterbag->get_set_node()->is_PBag());

      // Create a new Iter-bag.
      DBG_TRACE(DEBUG_BAGS, "frame %ld creates an Iter-bag ", func_id);
      func->create_iterbag();
      DBG_TRACE(DEBUG_BAGS, "%p\n", func->Iterbag);
    }
  }
}

void CilkSanImpl_t::do_loop_end() {
  DBG_TRACE(DEBUG_CALLBACK, "do_loop_end()\n");
  // frame_stack.head()->Sbag->get_node()->clear_version();
  FrameData_t *func = frame_stack.head();
  cilksan_assert(in_loop());
  // Get this frame's P-bag, creating it if necessary.
  DisjointSet_t<SPBagInterface *> *my_pbag = func->Pbag;
  if (!my_pbag) {
    DBG_TRACE(DEBUG_BAGS,
              "frame %ld creates a P-bag ",
              func->Sbag->get_set_node()->get_func_id());
    my_pbag =
      new DisjointSet_t<SPBagInterface *>(
          new PBag_t(func->Sbag->get_node()));
    func->set_pbag(my_pbag);
    DBG_TRACE(DEBUG_BAGS, "%p\n", my_pbag);
  }
  cilksan_assert(my_pbag && my_pbag->get_set_node()->is_PBag());

  // Combine the Iter-bag into this S-bag.
  if (func->is_Iterbag_used()) {
    DBG_TRACE(DEBUG_BAGS,
              "Merge Iter-bag in loop frame %ld into P-bag.\n",
              my_pbag->get_set_node()->get_func_id());
    my_pbag->combine(func->Iterbag);
    func->set_Pbag_used();
    cilksan_assert(func->Iterbag->get_set_node()->is_PBag());
  }
  // The loop frame is done, so clear the Iter-bag.
  func->set_iterbag(nullptr);

  // Return from the frame.
  do_leave_begin();
  do_leave_end();
}

void CilkSanImpl_t::do_sync_begin() {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_CALLBACK, "frame %ld cilk_sync_begin\n",
            frame_stack.head()->Sbag->get_node()->get_func_id());
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = CILK_SYNC);
}

void CilkSanImpl_t::do_sync_end() {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_CALLBACK, "cilk_sync_end\n");
  cilksan_assert(last_event == CILK_SYNC);
  WHEN_CILKSAN_DEBUG(last_event = NONE);
  complete_sync();
}

void CilkSanImpl_t::do_leave_begin() {
  cilksan_assert(CILKSAN_INITIALIZED);
  cilksan_assert(last_event == NONE);
  WHEN_CILKSAN_DEBUG(last_event = LEAVE_FRAME_OR_HELPER);
  DBG_TRACE(DEBUG_CALLBACK, "frame %ld cilk_leave_begin\n",
            frame_stack.head()->frame_data.frame_id);
  cilksan_assert(frame_stack.size() > 1);

  switch(frame_stack.head()->frame_data.entry_type) {
  case SPAWNER:
    DBG_TRACE(DEBUG_CALLBACK, "cilk_leave_frame_begin\n");
    break;
  case HELPER:
    DBG_TRACE(DEBUG_CALLBACK, "cilk_leave_helper_begin\n");
    break;
  case DETACHER:
    DBG_TRACE(DEBUG_CALLBACK, "cilk_leave_begin from detach\n");
    break;
  }

  if (DETACHER == frame_stack.head()->frame_data.entry_type)
    return_from_detach();
  else
    leave_cilk_function();
}

void CilkSanImpl_t::do_leave_end() {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_CALLBACK, "cilk_leave_end\n");
  cilksan_assert(last_event == LEAVE_FRAME_OR_HELPER);
  WHEN_CILKSAN_DEBUG(last_event = NONE);
}

// called by record_memory_read/write, with the access broken down into 64-byte
// aligned memory accesses
inline void
CilkSanImpl_t::record_mem_helper(bool is_read, const csi_id_t acc_id,
                                 uintptr_t addr, size_t mem_size,
                                 bool on_stack) {
  FrameData_t *f = frame_stack.head();
  bool write_in_shadow = shadow_memory.does_access_exists(false, addr, mem_size);
  bool read_in_shadow = shadow_memory.does_access_exists(true, addr, mem_size);

  // guaranteed safe if:
  //  1. it's a write, there are no other reads or writes
  //  2. it's a read, there are at most only reads
  if (!write_in_shadow && !read_in_shadow)
    shadow_memory.insert_access(is_read, acc_id, addr, mem_size, f,
                                call_stack);
  else {
    // we know this access can potentially lead to a race, so we check
    // check_races_and_update assumes both r/w are already present
    // but that's not true so there's a lot of casework
    if (is_read) {
      if (read_in_shadow) {
        if (!write_in_shadow)
          shadow_memory.update_with_read(acc_id, addr, mem_size, on_stack, f,
                                         call_stack);
        else
          check_races_and_update(is_read, acc_id, addr, mem_size, on_stack,
                                 f, call_stack, shadow_memory);
      } else {
        shadow_memory.insert_access(is_read, acc_id, addr, mem_size, f,
                                    call_stack);
        shadow_memory.check_race_with_prev_write(true, acc_id, addr, mem_size,
                                                 on_stack, f, call_stack);
      }
    } else {
      if (read_in_shadow && write_in_shadow) {
        check_races_and_update(is_read, acc_id, addr, mem_size, on_stack,
                               f, call_stack, shadow_memory);
      } else if (read_in_shadow) {
        shadow_memory.insert_access(is_read, acc_id, addr, mem_size, f,
                                    call_stack);
        shadow_memory.check_race_with_prev_read(acc_id, addr, mem_size,
                                                on_stack, f, call_stack);
      } else {
        shadow_memory.check_and_update_write(acc_id, addr, mem_size,
                                             on_stack, f, call_stack);
      }
    }
  }
}

void CilkSanImpl_t::do_read(const csi_id_t load_id,
                            uintptr_t addr, size_t mem_size) {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_MEMORY, "record read %lu: %lu bytes at addr %p and rip %p.\n",
            load_id, mem_size, addr, load_pc[load_id]);
  ++num_reads_checked;

  bool on_stack = is_on_stack(addr);
  if (on_stack)
    advance_stack_frame(addr);

  record_mem_helper(true, load_id, addr, mem_size, on_stack);
}

void CilkSanImpl_t::do_write(const csi_id_t store_id,
                             uintptr_t addr, size_t mem_size) {
  cilksan_assert(CILKSAN_INITIALIZED);
  DBG_TRACE(DEBUG_MEMORY, "record write %ld: %lu bytes at addr %p and rip %p.\n",
            store_id, mem_size, addr, store_pc[store_id]);
  ++num_writes_checked;

  bool on_stack = is_on_stack(addr);
  if (on_stack)
    advance_stack_frame(addr);

  record_mem_helper(false, store_id, addr, mem_size, on_stack);
}

// clear the memory block at [start,start+size) (end is exclusive).
void CilkSanImpl_t::clear_shadow_memory(size_t start, size_t size) {
  DBG_TRACE(DEBUG_MEMORY, "cilksan_clear_shadow_memory(%p, %ld)\n",
            start, size);
  shadow_memory.clear(start, size);
}

void CilkSanImpl_t::record_alloc(size_t start, size_t size,
                                 csi_id_t alloca_id) {
  DBG_TRACE(DEBUG_MEMORY, "cilksan_record_alloc(%p, %ld)\n",
            start, size);
  FrameData_t *f = frame_stack.head();
  shadow_memory.record_alloc(start, size, f, call_stack, alloca_id);
}

void CilkSanImpl_t::clear_alloc(size_t start, size_t size) {
  DBG_TRACE(DEBUG_MEMORY, "cilksan_clear_alloc(%p, %ld)\n",
            start, size);
  shadow_memory.clear_alloc(start, size);
}

inline void CilkSanImpl_t::print_stats() {
  // TODO: Add stats?
  // std::cout << "max sync block size seen: "
  //           << accounted_max_sync_block_size
  //           << "    (from user input: " << max_sync_block_size << ", average: "
  //           << ( (float)accum_sync_block_size / num_of_sync_blocks ) << ")"
  //           << std::endl;
  // std::cout << "max continuation depth seen: "
  //           << accounted_max_cont_depth << std::endl;
  std::cout << "number of reads checked:  " << num_reads_checked << "\n";
  std::cout << "number of writes checked: " << num_writes_checked << "\n";
}

void CilkSanImpl_t::deinit() {
  static bool deinit = false;
  // XXX: kind of a hack, but somehow this gets called twice.
  if (!deinit) deinit = true;
  else return; // deinit-ed already

  print_race_report();
  // Optionally print statistics.
  char *e = getenv("CILKSAN_STATS");
  if (e && 0 != strcmp(e, "0"))
    print_stats();

  cilksan_assert(frame_stack.size() == 1);

  shadow_memory.destruct();

  // Remove references to the disjoint set nodes so they can be freed.
  cilksan_assert(frame_stack.head()->Pbag == nullptr);
  frame_stack.head()->reset();
  frame_stack.pop();
  cilksan_assert(frame_stack.size() == 0);

  WHEN_CILKSAN_DEBUG({
      if (DisjointSet_t<SPBagInterface *>::debug_count != 0)
        std::cerr << "DisjointSet_t<SPBagInterface *>::debug_count = "
                  << DisjointSet_t<SPBagInterface *>::debug_count << "\n";
      if (SBag_t::debug_count != 0)
        std::cerr << "SBag_t::debug_count = "
                  << SBag_t::debug_count << "\n";
      if (PBag_t::debug_count != 0)
        std::cerr << "PBag_t::debug_count = "
                  << PBag_t::debug_count << "\n";
      // cilksan_assert(DisjointSet_t<SPBagInterface *>::debug_count == 0);
      // cilksan_assert(SBag_t::debug_count == 0);
      // cilksan_assert(PBag_t::debug_count == 0);
    });

  // Free the call-stack nodes in the free list.
  call_stack_node_t::cleanup_freelist();

  // // Free the disjoint-set nodes in the free list.
  // DisjointSet_t<SPBagInterface *>::cleanup_freelist();

  // Free the free lists for SBags and PBags.
  SBag_t::cleanup_freelist();
  PBag_t::cleanup_freelist();

  disjoint_set_list.free_list();

  // if(first_error != 0) exit(first_error);
}

void CilkSanImpl_t::init() {
  DBG_TRACE(DEBUG_CALLBACK, "cilksan_init()\n");
  std::cout << "Running Cilksan race detector\n";
  // std::cout << "cilksan_init() version 19\n";

  // cilksan_assert(stack_high_addr != 0 && stack_low_addr != 0);

  // these are true upon creation of the stack
  cilksan_assert(frame_stack.size() == 1);
  // cilksan_assert(entry_stack.size() == 1);
  // // actually only used for debugging of reducer race detection
  // WHEN_CILKSAN_DEBUG(rts_deque_begin = rts_deque_end = 1);

  shadow_memory.init();

  // for the main function before we enter the first Cilk context
  DisjointSet_t<SPBagInterface *> *sbag;
  DBG_TRACE(DEBUG_BAGS, "Creating SBag for frame %ld\n", frame_id);
  sbag = new DisjointSet_t<SPBagInterface *>(new SBag_t(frame_id, call_stack));
  cilksan_assert(sbag->get_set_node()->is_SBag());
  frame_stack.head()->set_sbag(sbag);
  WHEN_CILKSAN_DEBUG(frame_stack.head()->frame_data.frame_type = FULL_FRAME);
  WHEN_CILKSAN_DEBUG(CILKSAN_INITIALIZED = true);
}

extern "C" int __cilksan_error_count() {
  return CilkSanImpl.get_num_races_found();
}

// This funciton parse the input supplied to the user program and get the params
// meant for cilksan (everything after "--").  It return the index in which it
// found "--" so the user program knows when to stop parsing inputs.
extern "C" int __cilksan_parse_input(int argc, char *argv[]) {
  int i = 0;
  // uint32_t seed = 0;
  int stop = 0;

  while(i < argc) {
    if(!strncmp(argv[i], "--", strlen("--")+1)) {
      stop = i++;
      break;
    }
    i++;
  }

  while(i < argc) {
    char *arg = argv[i];
    // if(!strncmp(arg, "-cr", strlen("-cr")+1)) {
    //   i++;
    //   check_reduce = true;
    //   continue;

    // } else if(!strncmp(arg, "-update", strlen("-update")+1)) {
    //   i++;
    //   cont_depth_to_check = (uint64_t) atol(argv[i++]);
    //   continue;

    // } else if(!strncmp(arg, "-sb_size", strlen("-sb_size")+1)) {
    //   i++;
    //   max_sync_block_size = (uint32_t) atoi(argv[i++]);
    //   continue;

    // } else if(!strncmp(arg, "-s", strlen("-s")+1)) {
    //   i++;
    //   seed = (uint32_t) atoi(argv[i++]);
    //   continue;

    // } else if(!strncmp(arg, "-steal", strlen("-steal")+1)) {
    //   i++;
    //   cilksan_assert(steal_point1 < steal_point2
    //               && steal_point2 < steal_point3);
    //   check_reduce = true;
    //   continue;

    // } else {
      i++;
      std::cout << "Unrecognized input " << arg << ", ignore and continue."
                << std::endl;
    // }
  }

  // std::cout << "==============================================================="
  //           << std::endl;
  // if(cont_depth_to_check != 0) {
  //   check_reduce = false;
  //   std::cout << "This run will check updates for races with " << std::endl
  //             << "steals at continuation depth " << cont_depth_to_check;
  // } else if(check_reduce) {
  //   std::cout << "This run will check reduce functions for races " << std::endl
  //             << "with simulated steals ";
  //   if(max_sync_block_size > 1) {
  //     std::cout << "at randomly chosen continuation points \n"
  //               << "(assume block size "
  //               << max_sync_block_size << ")";
  //     if(seed) {
  //       std::cout << ", chosen using seed " << seed;
  //       srand(seed);
  //     } else {
  //       // srand(time(NULL));
  //     }
  //   } else {
  //     if(steal_point1 != steal_point2 && steal_point2 != steal_point3) {
  //       std::cout << "at steal points: " << steal_point1 << ", "
  //                 << steal_point2 << ", " << steal_point3 << ".";
  //     } else {
  //       simulate_all_steals = true;
  //       check_reduce = false;
  //       std::cout << "at every continuation point.";
  //     }
  //   }
  // } else {
  //   // cont_depth_to_check == 0 and check_reduce = false
  //   std::cout << "This run will check for races without simulated steals.";
  // }
  // std::cout << std::endl;
  // std::cout << "==============================================================="
  //           << std::endl;

  // cilksan_assert(!check_reduce || cont_depth_to_check == 0);
  // cilksan_assert(!check_reduce || max_sync_block_size > 1 || steal_point1 != steal_point2);

  return (stop == 0 ? argc : stop);
}

// XXX: Should really be in print_addr.cpp, but this will do for now
inline void CilkSanImpl_t::print_current_function_info() {
  FrameData_t *f = frame_stack.head();
  // std::cout << "steal points: " << f->steal_points[0] << ", "
  //           << f->steal_points[1] << ", " << f->steal_points[2] << std::endl;
  // std::cout << "curr sync block size: " << f->current_sync_block_size << std::endl;
  std::cout << "frame id: " << f->Sbag->get_node()->get_func_id() << std::endl;
}
