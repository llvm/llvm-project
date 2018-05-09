// -*- C++ -*-
#ifndef __CILKSAN_INTERNAL_H__
#define __CILKSAN_INTERNAL_H__

#include <cstdio>
#include <vector>
#include <map>
#include <iostream>

#include "csan.h"

#define UNINIT_STACK_PTR ((uintptr_t)0LL)
#define UNINIT_VIEW_ID ((uint64_t)0LL)

#include "cilksan.h"
#include "stack.h"

#define BT_OFFSET 1
#define BT_DEPTH 2

// The context in which the access is made; user = user code, update = update
// methods for a reducer object; reduce = reduce method for a reducer object
enum AccContextType_t { USER = 1, UPDATE = 2, REDUCE = 3 };
// W stands for write, R stands for read
enum RaceType_t { RW_RACE = 1, WW_RACE = 2, WR_RACE = 3 };

enum CallType_t { CALL, SPAWN };
// typedef std::pair<CallType_t, csi_id_t> CallID_t;
// typedef Stack_t<CallID_t> call_stack_t;
struct CallID_t {
  csi_id_t id;
  CallType_t type;
  CallID_t() : id(UNKNOWN_CSI_ID), type(CALL) {}

  CallID_t(CallType_t type, csi_id_t id)
      : id(id), type(type)
  {
    // if (SPAWN == type)
    //   // Assumes UNKNOWN_CSI_ID == -1
    //   id = -id - 3;
  }

  CallID_t(const CallID_t &copy) : id(copy.id), type(copy.type) {}

  CallID_t &operator=(const CallID_t &copy) {
    id = copy.id;
    type = copy.type;
    return *this;
  }

  CallType_t getType() const {
    // if (id < -1)
    //   return SPAWN;
    // return CALL;
    return type;
  }

  csi_id_t getID() const {
    // if (id < -1)
    //   // Assumes UNKNOWN_CSI_ID == -1
    //   return -id - 3;
    return id;
  }

  bool operator==(const CallID_t &that) {
    return (id == that.id) && (type == that.type);
  }

  bool operator!=(const CallID_t &that) {
    return !(*this == that);
  }

  inline friend
  std::ostream& operator<<(std::ostream &os, const CallID_t &id) {
    switch (id.type) {
    case CALL:
      os << "CALL " << id.id;
      break;
    case SPAWN:
      os << "SPAWN " << id.id;
      break;
    }
    return os;
  }
};

struct call_stack_node_t {
  CallID_t id;
  // std::shared_ptr<call_stack_node_t> prev;
  call_stack_node_t *prev;
  int64_t ref_count;
  call_stack_node_t(CallID_t id,
                    call_stack_node_t *prev = nullptr)
                    // std::shared_ptr<call_stack_node_t> prev = nullptr)
      // Set ref_count to 1, under the assumption that only call_stack_t
      // constructs nodes and will immediately set the tail pointer to point to
      // this node.
      : id(id), prev(prev), ref_count(1) {
    if (prev)
      prev->ref_count++;
  }

  ~call_stack_node_t() {
    // std::cerr << "call_stack_node_t destructing";
    // if (prev)
    //   std::cerr << "(PREV " << prev->id << " REF_COUNT " << prev->ref_count << ")";
    // std::cerr << std::endl;
    assert(!ref_count);
    // prev.reset();
    if (prev) {
      call_stack_node_t *old_prev = prev;
      prev = nullptr;
      assert(old_prev->ref_count > 0);
      old_prev->ref_count--;
      if (!old_prev->ref_count)
        delete old_prev;
    }
  }

  // Simple free-list allocator to conserve space and time in managing
  // call_stack_node_t objects.
  static call_stack_node_t *free_list;

  void *operator new(size_t size) {
    if (free_list) {
      call_stack_node_t *new_node = free_list;
      free_list = free_list->prev;
      return new_node;
    }
    return ::operator new(size);
  }

  void operator delete(void *ptr) {
    call_stack_node_t *del_node = reinterpret_cast<call_stack_node_t *>(ptr);
    del_node->prev = free_list;
    free_list = del_node;
  }

  static void cleanup_freelist() {
    call_stack_node_t *node = free_list;
    call_stack_node_t *next = nullptr;
    while (node) {
      next = node->prev;
      ::operator delete(node);
      node = next;
    }
  }
};

struct call_stack_t {
  // int size;
  // std::shared_ptr<call_stack_node_t> tail;
  call_stack_node_t *tail;
  call_stack_t() : // size(0),
                   tail(nullptr) {}

  call_stack_t(const call_stack_t &copy)
      : // size(copy.size),
        tail(copy.tail)
  {
    if (tail)
      tail->ref_count++;
  }

  ~call_stack_t() {
    // std::cerr << "call_stack_t destructing";
    // if (tail)
    //   std::cerr << " (TAIL " << tail->id << " REF_COUNT " << tail->ref_count << ")";
    // std::cerr << std::endl;
    // tail.reset();
    if (tail) {
      call_stack_node_t *old_tail = tail;
      tail = nullptr;
      assert(old_tail->ref_count > 0);
      old_tail->ref_count--;
      if (!old_tail->ref_count)
        delete old_tail;
    }
  }

  call_stack_t &operator=(const call_stack_t &copy) {
    // size = copy.size;
    call_stack_node_t *old_tail = tail;
    tail = copy.tail;
    if (tail)
      tail->ref_count++;
    if (old_tail) {
      old_tail->ref_count--;
      if (!old_tail->ref_count)
        delete old_tail;
    }
    return *this;
  }

  call_stack_t &operator=(const call_stack_t &&move) {
    // size = copy.size;
    if (tail) {
      tail->ref_count--;
      if (!tail->ref_count)
        delete tail;
    }
    tail = move.tail;
    return *this;
  }

  void overwrite(const call_stack_t &copy) {
    tail = copy.tail;
  }

  void push(CallID_t id) {
    // size++;
    // tail = std::make_shared<call_stack_node_t>(id, tail);
    call_stack_node_t *new_node = new call_stack_node_t(id, tail);
    // new_node has ref_count 1 and, if tail was not null, has incremented
    // tail's ref count.
    tail = new_node;
    if (tail->prev) {
      assert(tail->prev->ref_count > 1);
      tail->prev->ref_count--;
    }
    // now the ref_count's should reflect the pointer structure.
  }

  void pop() {
    assert(tail);
    // size--;
    call_stack_node_t *old_node = tail;
    tail = tail->prev;
    if (tail)
      tail->ref_count++;
    assert(old_node->ref_count > 0);
    old_node->ref_count--;
    if (!old_node->ref_count)
      // Deleting the old node will decrement tail's ref count.
      delete old_node;
  }

  int size() const {
    call_stack_node_t *node = tail;
    int size = 0;
    while (node) {
      ++size;
      node = node->prev;
    }
    return size;
  }
};

class AccessLoc_t {
public:
  csi_id_t acc_loc;
  // std::vector<CallID_t> call_stack;
  // const std::shared_ptr<call_stack_node_t> call_stack;
  call_stack_t call_stack;
  // int64_t ref_count;

  AccessLoc_t() : acc_loc(), call_stack()// , ref_count(1)
  {}

  AccessLoc_t(csi_id_t _acc_loc, const call_stack_t &_call_stack)
      : acc_loc(_acc_loc), call_stack(_call_stack)// , ref_count(1)
        // call_stack(_call_stack.size()-1)
  {
    // std::cerr << "AccessLoc_t constructing";
    // if (call_stack.tail)
    //   std::cerr << " (TAIL " << call_stack.tail->id << " REF_COUNT " << call_stack.tail->ref_count << ")";
    // std::cerr << std::endl;
    // for (int i = 0; i < _call_stack.size()-1; ++i)
    //   call_stack[i] = *_call_stack.at(i);
  }

  AccessLoc_t(const AccessLoc_t &copy)
      : acc_loc(copy.acc_loc), call_stack(copy.call_stack)// , ref_count(1)
  {
    // std::cerr << "AccessLoc_t copy-constructing";
    // if (call_stack.tail)
    //   std::cerr << " (TAIL " << call_stack.tail->id << " REF_COUNT " << call_stack.tail->ref_count << ")";
    // std::cerr << std::endl;
  }

  AccessLoc_t(const AccessLoc_t &&move)
      : acc_loc(move.acc_loc)// , ref_count(1)
  {
    call_stack.overwrite(move.call_stack);
    // std::cerr << "AccessLoc_t copy-constructing";
    // if (call_stack.tail)
    //   std::cerr << " (TAIL " << call_stack.tail->id << " REF_COUNT " << call_stack.tail->ref_count << ")";
    // std::cerr << std::endl;
  }

  ~AccessLoc_t() {
    // assert(ref_count <= 1);
    // std::cerr << "AccessLoc_t destructing";
    // if (call_stack.tail)
    //   std::cerr << " (TAIL " << call_stack.tail->id << " REF_COUNT " << call_stack.tail->ref_count << ")";
    // std::cerr << std::endl;
  }

  csi_id_t getID() const {
    return acc_loc;
  }

  AccessLoc_t& operator=(const AccessLoc_t &copy) {
    acc_loc = copy.acc_loc;
    call_stack = copy.call_stack;
    return *this;
  }

  AccessLoc_t& operator=(const AccessLoc_t &&move) {
    acc_loc = move.acc_loc;
    call_stack = std::move(move.call_stack);
    return *this;
  }

  int64_t inc_ref_count(int64_t count = 1) {
    // ref_count += count;
    // return ref_count;
    if (!call_stack.tail)
      return 0;
    call_stack.tail->ref_count += count;
    return call_stack.tail->ref_count;
  }

  int64_t dec_ref_count(int64_t count = 1) {
    // assert(ref_count >= count);
    // ref_count -= count;
    // if (!ref_count) {
    //   delete this;
    //   return 0;
    // }
    // return ref_count;
    if (!call_stack.tail)
      return 0;
    assert(call_stack.tail->ref_count >= count);
    call_stack.tail->ref_count -= count;
    if (!call_stack.tail->ref_count) {
      delete call_stack.tail;
      call_stack.tail = nullptr;
      return 0;
    }
    return call_stack.tail->ref_count;
  }

  bool operator==(const AccessLoc_t &that) const {
    if (acc_loc != that.acc_loc)
      return false;

    call_stack_node_t *this_node = call_stack.tail;
    call_stack_node_t *that_node = that.call_stack.tail;
    while (this_node && that_node) {
      if (this_node->id != that_node->id)
        return false;
      this_node = this_node->prev;
      that_node = that_node->prev;
    }
    if (this_node || that_node)
      return false;
    return true;
  }

  // Unsafe method!  Only use this if you know what you're doing.
  void overwrite(const AccessLoc_t &copy) {
    acc_loc = copy.acc_loc;
    call_stack.overwrite(copy.call_stack);
  }

  // Unsafe method!  Only use this if you know what you're doing.
  void clear() {
    call_stack.tail = nullptr;
  }

  // bool hasValidLoc() const {
  //   return nullptr != call_stack.tail;
  // }

  bool operator!=(const AccessLoc_t &that) const {
    return !(*this == that);
  }

  bool operator<(const AccessLoc_t &that) const {
    return acc_loc < that.acc_loc;
  }

  inline friend
  std::ostream& operator<<(std::ostream &os, const AccessLoc_t &loc) {
    os << loc.acc_loc;
    // std::shared_ptr<call_stack_node_t> node(loc.call_stack.tail);
    call_stack_node_t *node = loc.call_stack.tail;
    while (node) {
      switch (node->id.getType()) {
      case CALL:
        os << " CALL";
        break;
      case SPAWN:
        os << " SPAWN";
        break;
      }
      os << " " << std::dec << node->id.getID();
      node = node->prev;
    }
    return os;
  }
};

typedef struct RaceInfo_t {
  const AccessLoc_t first_inst;  // instruction addr of the first access
  const AccessLoc_t second_inst; // instruction addr of the second access
  const AccessLoc_t alloc_inst;  // instruction addr of memory allocation
  uintptr_t addr;          // addr of memory location that got raced on
  enum RaceType_t type;    // type of race

  RaceInfo_t(const AccessLoc_t &_first, AccessLoc_t &&_second,
             const AccessLoc_t &_alloc,
             uintptr_t _addr, enum RaceType_t _type)
      : first_inst(_first), second_inst(_second), alloc_inst(_alloc),
        addr(_addr), type(_type)
  {
    // std::cerr << "RaceInfo_t constructing (" << first_inst << ", " << second_inst << ")\n";
  }

  ~RaceInfo_t() {
    // std::cerr << "RaceInfo_t destructing (" << first_inst << ", " << second_inst << ")\n";
  }

  bool is_equivalent_race(const struct RaceInfo_t& other) const {
    /*
    if( (type == other.type &&
         first_inst == other.first_inst && second_inst == other.second_inst) ||
        (first_inst == other.second_inst && second_inst == other.first_inst &&
         ((type == RW_RACE && other.type == WR_RACE) ||
          (type == WR_RACE && other.type == RW_RACE))) ) {
      return true;
    } */
    // Angelina: It turns out that, Cilkscreen does not care about the race
    // types.  As long as the access instructions are the same, it's considered
    // as a duplicate.
    if (((first_inst == other.first_inst && second_inst == other.second_inst) ||
         (first_inst == other.second_inst && second_inst == other.first_inst)) &&
        alloc_inst == other.alloc_inst) {
      return true;
    }
    return false;
  }
} RaceInfo_t;

// defined in print_addr.cpp
void report_race(const AccessLoc_t &first_inst, AccessLoc_t &&second_inst,
                 uintptr_t addr, enum RaceType_t race_type);
void report_race(const AccessLoc_t &first_inst, AccessLoc_t &&second_inst,
                 const AccessLoc_t &alloc_inst,
                 uintptr_t addr, enum RaceType_t race_type);

// public functions
void cilksan_init();
void cilksan_deinit();
void cilksan_do_enter_begin();
void cilksan_do_enter_helper_begin();
void cilksan_do_enter_end(uintptr_t stack_ptr);
void cilksan_do_detach_begin();
void cilksan_do_detach_end();
void cilksan_do_sync_begin();
void cilksan_do_sync_end();
void cilksan_do_return();
void cilksan_do_leave_begin();
void cilksan_do_leave_end();
void cilksan_do_leave_stolen_callback();

void cilksan_do_read(const csi_id_t load_id, uintptr_t addr, size_t len);
void cilksan_do_write(const csi_id_t store_id, uintptr_t addr, size_t len);
void cilksan_clear_shadow_memory(size_t start, size_t end);
void cilksan_record_alloc(size_t start, size_t end, csi_id_t alloca_id);
// void cilksan_do_function_entry(uint64_t an_address);
// void cilksan_do_function_exit();
#endif // __CILKSAN_INTERNAL_H__
