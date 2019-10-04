// -*- C++ -*-
#ifndef __CILKSAN_INTERNAL_H__
#define __CILKSAN_INTERNAL_H__

#include <cstdio>
#include <iostream>
#include <unordered_map>

#include "csan.h"

#define UNINIT_VIEW_ID ((uint64_t)0LL)

#include "cilksan.h"
#include "dictionary.h"
#include "disjointset.h"
#include "frame_data.h"
#include "shadow_mem.h"
#include "shadow_mem_allocator.h"
#include "stack.h"

#define BT_OFFSET 1
#define BT_DEPTH 2

// Top-level class implementing the tool.
class CilkSanImpl_t {
public:
  CilkSanImpl_t() {}
  ~CilkSanImpl_t();

  MALineAllocator &getMALineAllocator(unsigned Idx) {
    return MAAlloc[Idx];
  }

  using DJSAllocator = DisjointSet_t<SPBagInterface *>::DJSAllocator;
  DJSAllocator &getDJSAllocator() {
    return DJSAlloc;
  }

  // Initialization
  void init();
  void deinit();

  // Control-flow actions
  inline void record_call(const csi_id_t id, enum CallType_t ty) {
    call_stack.push(CallID_t(ty, id));
  }

  inline void record_call_return(const csi_id_t id, enum CallType_t ty) {
    assert(call_stack.tailMatches(CallID_t(ty, id)) &&
           "Mismatched hooks around call/spawn site");
    call_stack.pop();
  }

  // TODO: Fix this architecture-specific detail.
  static const uintptr_t STACK_ALIGN = 16;
#define NEXT_STACK_ALIGN(addr) \
  ((uintptr_t) ((addr - (STACK_ALIGN-1)) & (~(STACK_ALIGN-1))))
#define PREV_STACK_ALIGN(addr) (addr + STACK_ALIGN)

  inline void push_stack_frame(uintptr_t bp, uintptr_t sp) {
    DBG_TRACE(DEBUG_BASIC, "push_stack_frame %p--%p\n", bp, sp);
    // Record high location of the stack for this frame.
    uintptr_t high_stack = bp;
    // uintptr_t high_stack = PREV_STACK_ALIGN(bp);
    // fprintf(stderr, "bp = %p, NEXT_STACK_ALIGN = %p, PREV_STACK_ALIGN = %p\n",
    //         bp, NEXT_STACK_ALIGN(bp), PREV_STACK_ALIGN(bp));

    // if (sp_stack.size() > 1) {
    //   uintptr_t prev_stack = *sp_stack.head();

    //   // fprintf(stderr, "  NEXT_STACK_ALIGN(prev_stack) = %p\n",
    //   //         NEXT_STACK_ALIGN(prev_stack));
    //   if (high_stack > NEXT_STACK_ALIGN(prev_stack))
    //     high_stack = NEXT_STACK_ALIGN(prev_stack);
    //   // low_stack = (low_stack < (uintptr_t)sp) ? sp : low_stack;
    // }

    sp_stack.push();
    *sp_stack.head() = high_stack;
    // Record low location of the stack for this frame.  This value will be
    // updated by reads and writes to the stack.
    sp_stack.push();
    *sp_stack.head() = sp;
  }

  inline void advance_stack_frame(uintptr_t addr) {
    DBG_TRACE(DEBUG_BASIC, "advance_stack_frame %p to include %p\n",
              *sp_stack.head(), addr);
    if (addr < *sp_stack.head()) {
      *sp_stack.head() = addr;
      FrameData_t *cilk_func = frame_stack.head();
      cilk_func->Sbag->get_node()->set_rsp(addr);
    }
  }

  inline void pop_stack_frame() {
    // Pop stack pointers.
    uintptr_t low_stack = *sp_stack.head();
    sp_stack.pop();
    uintptr_t high_stack = *sp_stack.head();
    sp_stack.pop();
    DBG_TRACE(DEBUG_BASIC, "pop_stack_frame %p--%p\n", high_stack, low_stack);
    assert(low_stack <= high_stack);
    // Clear shadow memory of stack locations.  This seems to be necessary right
    // now, in order to handle functions that dynamically allocate stack memory.
    // if (high_stack != low_stack) {
      clear_shadow_memory(low_stack, high_stack - low_stack + 1);
      clear_alloc(low_stack, high_stack - low_stack + 1);
    // }
  }

  inline bool is_local_synced() const {
    FrameData_t *f = frame_stack.head();
    if (f->Pbags)
      for (unsigned i = 0; i < f->num_Pbags; ++i)
        if (f->Pbags[i])
          return false;
    return true;
  }

  void do_enter_begin(unsigned num_sync_reg);
  void do_enter_helper_begin(unsigned num_sync_reg);
  void do_enter_end(uintptr_t stack_ptr);
  void do_detach_begin();
  void do_detach_end();
  void do_loop_begin() {
    start_new_loop = true;
  }
  void do_loop_iteration_begin(uintptr_t stack_ptr, unsigned num_sync_reg);
  void do_loop_iteration_end();
  void do_loop_end(unsigned sync_reg);
  bool in_loop() const {
    return LOOP_FRAME == frame_stack.head()->frame_data.frame_type;
  }
  bool handle_loop() const {
    return in_loop() || (true == start_new_loop);
  }
  void do_sync_begin();
  void do_sync_end(unsigned sync_reg);
  void do_return();
  void do_leave_begin(unsigned sync_reg);
  void do_leave_end();
  // void do_function_entry(uint64_t an_address);
  // void do_function_exit();

  // Memory actions
  void do_read(const csi_id_t load_id, uintptr_t addr, size_t len);
  void do_write(const csi_id_t store_id, uintptr_t addr, size_t len);
  void clear_shadow_memory(size_t start, size_t end);
  void record_alloc(size_t start, size_t end, csi_id_t alloca_id);
  void clear_alloc(size_t start, size_t end);

  const call_stack_t &get_current_call_stack() const {
    return call_stack;
  }

  // defined in print_addr.cpp
  void report_race(
      const AccessLoc_t &first_inst, const AccessLoc_t &second_inst,
      uintptr_t addr, enum RaceType_t race_type);
  void report_race(
      const AccessLoc_t &first_inst, const AccessLoc_t &second_inst,
      const AccessLoc_t &alloc_inst, uintptr_t addr,
      enum RaceType_t race_type);
  void print_race_report();
  int get_num_races_found();
  void print_current_function_info();

private:
  inline void merge_bag_from_returning_child(bool returning_from_detach,
                                             unsigned sync_reg);
  inline void start_new_function(unsigned num_sync_reg);
  inline void exit_function();
  inline void enter_cilk_function(unsigned num_sync_reg);
  inline void leave_cilk_function(unsigned sync_reg);
  inline void enter_detach_child(unsigned num_sync_reg);
  inline void return_from_detach(unsigned sync_reg);
  inline void complete_sync(unsigned sync_reg);
  template<bool is_read>
  inline void record_mem_helper(const csi_id_t acc_id,
                                uintptr_t addr, size_t mem_size,
                                bool on_stack);
  // inline void print_current_function_info();
  inline void print_stats();

  // ANGE: Each function that causes a Disjoint set to be created has a
  // unique ID (i.e., Cilk function and spawned C function).
  // If a spawned function is also a Cilk function, a Disjoint Set is created
  // for code between the point where detach finishes and the point the Cilk
  // function calls enter_frame, which may be unnecessary in some case.
  // (But potentially necessary in the case where the base case is executed.)
  uint64_t frame_id = 0;

  // Data associated with the stack of Cilk frames or spawned C frames.
  // head contains the SP bags for the function we are currently processing
  Stack_t<FrameData_t> frame_stack;
  call_stack_t call_stack;
  Stack_t<uintptr_t> sp_stack;

  bool start_new_loop = false;

  // Shadow memory, which maps a memory address to its last reader and writer
  // and allocation.
  Shadow_Memory shadow_memory;

  // Use separate allocators for each dictionary in the shadow memory.
  MALineAllocator MAAlloc[3];

  // Allocator for disjoint sets
  DJSAllocator DJSAlloc;

  // A map keeping track of races found, keyed by the larger instruction address
  // involved in the race.  Races that have same instructions that made the same
  // types of accesses are considered as the the same race (even for races where
  // one is read followed by write and the other is write followed by read, they
  // are still considered as the same race).  Races that have the same
  // instruction addresses but different address for memory location is
  // considered as a duplicate.  The value of the map stores the number
  // duplicates for the given race.
  using RaceMap_t = std::unordered_multimap<uint64_t, RaceInfo_t>;
  RaceMap_t races_found;
  // The number of duplicated races found
  uint32_t duplicated_races = 0;

  // Basic statistics
  uint64_t num_reads_checked = 0;
  uint64_t num_writes_checked = 0;
};

#endif // __CILKSAN_INTERNAL_H__
