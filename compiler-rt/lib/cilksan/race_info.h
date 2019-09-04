// -*- C++ -*-
#ifndef __RACE_INFO_H__
#define __RACE_INFO_H__

#ifndef CHECK_EQUIVALENT_STACKS
#define CHECK_EQUIVALENT_STACKS false
#endif

// The context in which the access is made; user = user code, update = update
// methods for a reducer object; reduce = reduce method for a reducer object
enum AccContextType_t { USER = 1, UPDATE = 2, REDUCE = 3 };
// W stands for write, R stands for read
enum RaceType_t { RW_RACE = 1, WW_RACE = 2, WR_RACE = 3 };

// Class for representing a frame on the call stack.
enum CallType_t : uint8_t { CALL, SPAWN, LOOP };
class CallID_t {
  // We assume that the top 16 bits of a CSI ID, after the sign bit, are unused.
  // Since CSI ID's identify instructions in the program, this assumption bounds
  // the number of CSI ID's based on the size of virtual memory.
  static constexpr unsigned TYPE_SHIFT = 48;
  static constexpr csi_id_t ID_MASK = ((1UL << TYPE_SHIFT) - 1);
  static constexpr csi_id_t UNKNOWN_CSI_CALL_ID = UNKNOWN_CSI_ID & ID_MASK;

  csi_id_t typed_id;

public:
  CallID_t() : typed_id((static_cast<csi_id_t>(CALL) << TYPE_SHIFT) |
                        UNKNOWN_CSI_CALL_ID) {}
  CallID_t(CallType_t type, csi_id_t id)
      : typed_id((static_cast<csi_id_t>(type) << TYPE_SHIFT) | (id & ID_MASK))
  {}
  CallID_t(const CallID_t &copy) : typed_id(copy.typed_id) {}

  inline CallID_t &operator=(const CallID_t &copy) {
    typed_id = copy.typed_id;
    return *this;
  }

  inline CallType_t getType() const {
    return static_cast<CallType_t>(typed_id >> TYPE_SHIFT);
  }

  inline csi_id_t getID() const {
    return (typed_id & ID_MASK);
  }

  inline bool isUnknownID() const {
    return getID() == UNKNOWN_CSI_CALL_ID;
  }

  inline bool operator==(const CallID_t &that) const {
    return typed_id == that.typed_id;
  }
  inline bool operator!=(const CallID_t &that) const {
    return !(*this == that);
  }

  inline friend
  std::ostream& operator<<(std::ostream &os, const CallID_t &id) {
    switch (id.getType()) {
    case CALL:
      os << "CALL " << id.getID();
      break;
    case SPAWN:
      os << "SPAWN " << id.getID();
      break;
    case LOOP:
      os << "LOOP " << id.getID();
      break;
    }
    return os;
  }
};

// Specialized stack data structure for representing the call stack.  CilkSan
// models the call stack using a singly-linked list with reference-counted
// nodes.  When a race is recorded, the call stack for that race is preserved by
// saving a pointer to the tail of the call stack.

// Class for reference-counted nodes on the call stack.
class call_stack_node_t {
  friend class call_stack_t;
  friend class AccessLoc_t;

  CallID_t id;
  call_stack_node_t *prev;
  int64_t ref_count;

public:
  call_stack_node_t(CallID_t id, call_stack_node_t *prev = nullptr)
      // Set ref_count to 1, under the assumption that only call_stack_t
      // constructs nodes and will immediately set the tail pointer to point to
      // this node.
      : id(id), prev(prev), ref_count(1) {
    if (prev)
      prev->ref_count++;
  }

  ~call_stack_node_t() {
    cilksan_assert(!ref_count);
    if (prev) {
      call_stack_node_t *old_prev = prev;
      prev = nullptr;
      cilksan_assert(old_prev->ref_count > 0);
      old_prev->ref_count--;
      if (!old_prev->ref_count)
        delete old_prev;
    }
  }

  inline const CallID_t &getCallID() const {
    return id;
  }

  inline const call_stack_node_t *getPrev() const {
    return prev;
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

// Top-level class for the call stack.
class call_stack_t {
  friend class AccessLoc_t;

  // int size;
  call_stack_node_t *tail = nullptr;

public:
  call_stack_t() {}
  call_stack_t(const call_stack_t &copy)
      : // size(copy.size),
        tail(copy.tail)
  {
    if (tail)
      tail->ref_count++;
  }

  ~call_stack_t() {
    if (tail) {
      call_stack_node_t *old_tail = tail;
      tail = nullptr;
      cilksan_assert(old_tail->ref_count > 0);
      old_tail->ref_count--;
      if (!old_tail->ref_count)
        delete old_tail;
    }
  }

  inline const call_stack_node_t *getTail() const {
    return tail;
  }

  inline bool tailMatches(const CallID_t &id) const {
    return tail->id == id;
  }

  inline call_stack_t &operator=(const call_stack_t &copy) {
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

  inline call_stack_t &operator=(const call_stack_t &&move) {
    // size = copy.size;
    if (tail) {
      tail->ref_count--;
      if (!tail->ref_count)
        delete tail;
    }
    tail = move.tail;
    return *this;
  }

  inline void overwrite(const call_stack_t &copy) {
    tail = copy.tail;
  }

  inline void push(CallID_t id) {
    // size++;
    call_stack_node_t *new_node = new call_stack_node_t(id, tail);
    // new_node has ref_count 1 and, if tail was not null, has incremented
    // tail's ref count.
    tail = new_node;
    if (tail->prev) {
      cilksan_assert(tail->prev->ref_count > 1);
      tail->prev->ref_count--;
    }
    // now the ref_count's should reflect the pointer structure.
  }

  inline void pop() {
    cilksan_assert(tail);
    // size--;
    call_stack_node_t *old_node = tail;
    tail = tail->prev;
    if (tail)
      tail->ref_count++;
    cilksan_assert(old_node->ref_count > 0);
    old_node->ref_count--;
    if (!old_node->ref_count)
      // Deleting the old node will decrement tail's ref count.
      delete old_node;
  }

  inline int size() const {
    call_stack_node_t *node = tail;
    int size = 0;
    while (node) {
      ++size;
      node = node->prev;
    }
    return size;
  }
};

// Class representing a memory access.
class AccessLoc_t {
  // CSI ID of the access.
  csi_id_t acc_loc;

  // Call stack for the access.
  call_stack_t call_stack;
  // int64_t ref_count;

public:
  AccessLoc_t() : acc_loc(UNKNOWN_CSI_ID), call_stack()// , ref_count(1)
  {}

  AccessLoc_t(csi_id_t _acc_loc, const call_stack_t &_call_stack)
      : acc_loc(_acc_loc), call_stack(_call_stack)// , ref_count(1)
        // call_stack(_call_stack.size()-1)
  {}

  AccessLoc_t(const AccessLoc_t &copy)
      : acc_loc(copy.acc_loc), call_stack(copy.call_stack)// , ref_count(1)
  {}

  AccessLoc_t(AccessLoc_t &&move)
      : acc_loc(move.acc_loc)// , ref_count(1)
  {
    call_stack.overwrite(move.call_stack);
  }

  ~AccessLoc_t() {}

  inline csi_id_t getID() const {
    return acc_loc;
  }

  inline const call_stack_node_t *getCallStack() const {
    return call_stack.tail;
  }

  inline int getCallStackSize() const {
    return call_stack.size();
  }

  inline bool isValid() const {
    return acc_loc != UNKNOWN_CSI_ID;
  }

  inline void invalidate() {
    dec_ref_count();
    call_stack.tail = nullptr;
    acc_loc = UNKNOWN_CSI_ID;
  }

  inline AccessLoc_t& operator=(const AccessLoc_t &copy) {
    acc_loc = copy.acc_loc;
    call_stack = copy.call_stack;
    return *this;
  }

  inline AccessLoc_t& operator=(AccessLoc_t &&move) {
    acc_loc = move.acc_loc;
    call_stack = std::move(move.call_stack);
    return *this;
  }

  inline int64_t inc_ref_count(int64_t count = 1) {
    if (!call_stack.tail)
      return 0;
    call_stack.tail->ref_count += count;
    return call_stack.tail->ref_count;
  }

  inline int64_t dec_ref_count(int64_t count = 1) {
    if (!call_stack.tail)
      return 0;
    cilksan_assert(call_stack.tail->ref_count >= count);
    call_stack.tail->ref_count -= count;
    if (!call_stack.tail->ref_count) {
      delete call_stack.tail;
      call_stack.tail = nullptr;
      return 0;
    }
    return call_stack.tail->ref_count;
  }

  inline bool operator==(const AccessLoc_t &that) const {
    if (acc_loc != that.acc_loc)
      return false;
#if CHECK_EQUIVALENT_STACKS
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
#endif // CHECK_EQUIVALENT_STACKS
    return true;
  }

  // Unsafe method!  Only use this if you know what you're doing.
  inline void overwrite(const AccessLoc_t &copy) {
    acc_loc = copy.acc_loc;
    call_stack.overwrite(copy.call_stack);
  }

  // Unsafe method!  Only use this if you know what you're doing.
  inline void clear() {
    call_stack.tail = nullptr;
  }

  // bool hasValidLoc() const {
  //   return nullptr != call_stack.tail;
  // }

  inline bool operator!=(const AccessLoc_t &that) const {
    return !(*this == that);
  }

  inline bool operator<(const AccessLoc_t &that) const {
    return acc_loc < that.acc_loc;
  }

  inline friend
  std::ostream& operator<<(std::ostream &os, const AccessLoc_t &loc) {
    os << loc.acc_loc;
    // call_stack_node_t *node = loc.call_stack.tail;
    const call_stack_node_t *node = loc.getCallStack();
    while (node) {
      switch (node->getCallID().getType()) {
      case CALL:
        os << " CALL";
        break;
      case SPAWN:
        os << " SPAWN";
        break;
      case LOOP:
        os << " LOOP";
        break;
      }
      os << " " << std::dec << node->getCallID().getID();
      node = node->getPrev();
    }
    return os;
  }
};

// Class representing a single race.
class RaceInfo_t {
  // const AccessLoc_t first_inst;  // instruction addr of the first access
  // const AccessLoc_t second_inst; // instruction addr of the second access
  // const AccessLoc_t alloc_inst;  // instruction addr of memory allocation
  csi_id_t first_id;
  csi_id_t second_id;
  csi_id_t alloc_id;
  uintptr_t addr;          // addr of memory location that got raced on
  enum RaceType_t type;    // type of race

public:
  RaceInfo_t(const AccessLoc_t &_first, const AccessLoc_t &_second,
             const AccessLoc_t &_alloc, uintptr_t _addr, enum RaceType_t _type)
      : first_id(_first.getID()), second_id(_second.getID()),
        alloc_id(_alloc.getID()), addr(_addr), type(_type)
  {}

  ~RaceInfo_t() {}

  bool is_equivalent_race(const RaceInfo_t& other) const {
    /*
    if( (type == other.type &&
         first_inst == other.first_inst && second_inst == other.second_inst) ||
        (first_inst == other.second_inst && second_inst == other.first_inst &&
         ((type == RW_RACE && other.type == WR_RACE) ||
          (type == WR_RACE && other.type == RW_RACE))) ) {
      return true;
    } */
    // Angelina: It turns out that Cilkscreen does not care about the race
    // types.  As long as the access instructions are the same, it's considered
    // as a duplicate.
    if (((first_id == other.first_id && second_id == other.second_id) ||
         (first_id == other.second_id && second_id == other.first_id)) &&
        alloc_id == other.alloc_id) {
      return true;
    }
    return false;
  }

  inline void print(const AccessLoc_t &first, const AccessLoc_t &second,
                    const AccessLoc_t &alloc) const;
};

#endif  // __RACE_INFO_H__
