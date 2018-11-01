// -*- C++ -*-
#ifndef __DICTIONARY__
#define __DICTIONARY__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <map>

#include <execinfo.h>
#include <inttypes.h>

#include "debug_util.h"
#include "disjointset.h"
#include "mem_access.h"
#include "race_info.h"
#include "spbag.h"

class MemoryAccess_t {
public:
  DisjointSet_t<SPBagInterface *> *func = nullptr;
  AccessLoc_t loc;

  MemoryAccess_t()
      : func(nullptr), loc() { }

  MemoryAccess_t(DisjointSet_t<SPBagInterface *> *func,
                 csi_id_t acc_id, const call_stack_t &call_stack)
      : func(func), loc(AccessLoc_t(acc_id, call_stack)) {
    if (func)
      func->inc_ref_count();
  }

  MemoryAccess_t(const MemoryAccess_t &copy)
      : func(copy.func), loc(copy.loc) {
    if (func)
      func->inc_ref_count();
    // if (loc)
    //   loc->inc_ref_count();
  }

  MemoryAccess_t(const MemoryAccess_t &&move)
      : func(move.func), loc(std::move(move.loc)) {}

  ~MemoryAccess_t() {
    if (func)
      func->dec_ref_count();
    // if (loc)
    //   loc->dec_ref_count();
  }

  bool isValid() const {
    if (nullptr == func)
      assert(nullptr == loc.getCallStack());
    return (nullptr != func);
  }

  void invalidate() {
    if (func)
      func->dec_ref_count();
    func = nullptr;
    loc.dec_ref_count();
    // if (loc)
    //   loc->dec_ref_count();
    // loc = nullptr;
  }

  DisjointSet_t<SPBagInterface *> *getFunc() const {
    return func;
  }

  const AccessLoc_t &getLoc() const {
    return loc;
  }

  MemoryAccess_t &operator=(const MemoryAccess_t &copy) {
    if (func != copy.func) {
      if (copy.func)
        copy.func->inc_ref_count();
      if (func)
        func->dec_ref_count();
      func = copy.func;
    }
    loc = copy.loc;
    // if (loc != copy.loc) {
    //   if (copy.loc)
    //     copy.loc->inc_ref_count();
    //   if (loc)
    //     loc->dec_ref_count();
    //   loc = copy.loc;
    // }
    return *this;
  }

  MemoryAccess_t &operator=(const MemoryAccess_t &&move) {
    if (func)
      func->dec_ref_count();
    // if (loc)
    //   loc->dec_ref_count();
    func = move.func;
    loc = std::move(move.loc);
    return *this;
  }

  void inc_ref_counts(int64_t count) {
    assert(func);
    // assert(loc);
    func->inc_ref_count(count);
    loc.inc_ref_count(count);
    // loc->inc_ref_count(count);
  }

  void dec_ref_counts(int64_t count) {
    if (!func->dec_ref_count(count))
      func = nullptr;
    loc.dec_ref_count(count);
    // if (!loc->dec_ref_count(count))
    //   loc = nullptr;
  }

  void inherit(const MemoryAccess_t &copy) {
    if (func)
      func->dec_ref_count();
    func = copy.func;
    loc.dec_ref_count();
    // if (loc)
    //   loc->dec_ref_count();
    loc = copy.loc;
  }

  // Unsafe method!  Only use this if you know what you're doing.
  void overwrite(const MemoryAccess_t &copy) {
    func = copy.func;
    loc.overwrite(copy.loc);
  }

  // Unsafe method!  Only use this if you know what you're doing.
  void clear() {
    func = nullptr;
    loc.clear();
  }

  bool sameAccessLocPtr(const MemoryAccess_t &that) const {
    return loc.getCallStack() == that.loc.getCallStack();
  }

  // TODO: Get rid of PC from these comparisons
  bool operator==(const MemoryAccess_t &that) const {
    return (func == that.func); // && (loc == that.loc);
  }

  bool operator!=(const MemoryAccess_t &that) const {
    // return (func != that.func) || (loc != that.loc);
    return !(*this == that);
  }

  inline friend
  std::ostream& operator<<(std::ostream &os, const MemoryAccess_t &acc) {
    os << "function " << acc.func->get_node()->get_func_id() <<
      ", " << acc.loc;
    return os;
  }

  // Simple free-list allocator to conserve space and time in managing
  // arrays of PAGE_SIZE MemoryAccess_t objects.
  struct FreeNode_t {
    static size_t FreeNode_ObjSize;
    FreeNode_t *next;
  };
  static FreeNode_t *free_list;

  void *operator new[](size_t size) {
    if (!FreeNode_t::FreeNode_ObjSize)
      FreeNode_t::FreeNode_ObjSize = size;
    if (free_list) {
      assert(size == FreeNode_t::FreeNode_ObjSize);
      FreeNode_t *new_node = free_list;
      free_list = free_list->next;
      return new_node;
    }
    // std::cerr << "MemoryAccess_t::new[] called, size " << size << "\n";
    return ::operator new[](size);
  }

  void operator delete[](void *ptr) {
    FreeNode_t *del_node = reinterpret_cast<FreeNode_t *>(ptr);
    del_node->next = free_list;
    free_list = del_node;
  }

  static void cleanup_freelist() {
    FreeNode_t *node = free_list;
    FreeNode_t *next = nullptr;
    while (node) {
      next = node->next;
      ::operator delete[](node);
      node = next;
    }
  }

};

// typedef DisjointSet_t<SPBagInterface *> * value_type00;
typedef MemoryAccess_t value_type00;
// struct value_type00 {
//   std::shared_ptr<MemoryAccess_t> val;

//   value_type00() : val(nullptr) {}

//   value_type00(MemoryAccess_t acc)
//       : val(std::make_shared<MemoryAccess_t>(acc))
//   {}

//   value_type00(const value_type00 &copy)
//       : val(copy.val)
//   {}

//   value_type00(const value_type00 &&move)
//       : val(std::move(move.val))
//   {}

//   value_type00 &operator=(const value_type00 &copy) {
//     val = copy.val;
//     return *this;
//   }

//   value_type00 &operator=(const value_type00 &&move) {
//     val = std::move(move.val);
//     return *this;
//   }

//   bool isValid() const {
//     return (bool)val && val->isValid();
//   }

//   void invalidate() {
//     return val.reset();
//   }

//   bool operator==(const value_type00 &that) const {
//     if (val == that.val)
//       return true;
//     if (((bool)val && !(bool)that.val) ||
//         (!(bool)val && (bool)that.val))
//       return false;
//     return *val == *that.val;
//   }

//   bool operator!=(const value_type00 &that) const {
//     return !(val == that.val);
//   }
// };

class Dictionary {
public:
  static const value_type00 null_val;

  virtual value_type00 *find(uint64_t key) {
    return nullptr;
  }

  virtual value_type00 *find_group(uint64_t key, size_t max_size,
                                   size_t &num_elems) {
    num_elems = 1;
    return nullptr;
  }

  virtual value_type00 *find_exact_group(uint64_t key, size_t max_size,
                                         size_t &num_elems) {
    num_elems = 1;
    return nullptr;
  }

  virtual const value_type00 &operator[] (uint64_t key) {
    return null_val;
  }

  virtual void erase(uint64_t key) {}

  virtual void erase(uint64_t key, size_t size) {}

  virtual bool includes(uint64_t key) {
    return false;
  }

  virtual bool includes(uint64_t key, size_t size) {
    return false;
  }

  virtual void insert(uint64_t key, const value_type00 &f) {}

  virtual void insert(uint64_t key, size_t size, const value_type00 &f) {}
  virtual void set(uint64_t key, size_t size, value_type00 &&f) {
    insert(key, size, std::move(f));
  }
  virtual void insert_into_found_group(uint64_t key, size_t size,
                                       value_type00 *dst,
                                       value_type00 &&f) {
    insert(key, size, std::move(f));
  }

  virtual ~Dictionary() {};

  //uint32_t run_length(uint64_t key) {return 0;}

  //virtual void destruct() {}
};

#endif  // __DICTIONARY__
