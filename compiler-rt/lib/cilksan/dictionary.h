// -*- C++ -*-
#ifndef __DICTIONARY__
#define __DICTIONARY__

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <inttypes.h>

#include "debug_util.h"
#include "disjointset.h"
#include "race_info.h"
#include "spbag.h"

class MemoryAccess_t {
  static constexpr unsigned VERSION_SHIFT = 48;
  static constexpr csi_id_t ID_MASK = ((1UL << VERSION_SHIFT) - 1);
  static constexpr csi_id_t UNKNOWN_CSI_ACC_ID = UNKNOWN_CSI_ID & ID_MASK;
public:
  DisjointSet_t<SPBagInterface *> *func = nullptr;
  // AccessLoc_t loc;
  csi_id_t ver_acc_id = UNKNOWN_CSI_ACC_ID;

  MemoryAccess_t() {}
  MemoryAccess_t(DisjointSet_t<SPBagInterface *> *func, csi_id_t acc_id)
      : func(func), ver_acc_id((acc_id & ID_MASK))
  {
    if (func) {
      func->inc_ref_count();
      ver_acc_id |=
        static_cast<csi_id_t>(func->get_node()->get_version()) << VERSION_SHIFT;
    }
  }

  MemoryAccess_t(const MemoryAccess_t &copy)
      : func(copy.func),
        // loc(copy.loc)
        ver_acc_id(copy.ver_acc_id) {
    if (func)
      func->inc_ref_count();
    // if (loc)
    //   loc->inc_ref_count();
  }

  MemoryAccess_t(const MemoryAccess_t &&move)
      : func(move.func),
        // loc(std::move(move.loc))
        ver_acc_id(move.ver_acc_id) {}

  ~MemoryAccess_t() {
    if (func) {
      func->dec_ref_count();
      func = nullptr;
    }
    // if (loc)
    //   loc->dec_ref_count();
  }

  bool isValid() const {
    // if (nullptr == func)
    //   assert(nullptr == loc.getCallStack());
    return (nullptr != func);
  }

  void invalidate() {
    if (func)
      func->dec_ref_count();
    func = nullptr;
    // loc.invalidate();
    ver_acc_id = UNKNOWN_CSI_ACC_ID;
    // loc.dec_ref_count();
    // if (loc)
    //   loc->dec_ref_count();
    // loc = nullptr;
  }

  DisjointSet_t<SPBagInterface *> *getFunc() const {
    return func;
  }

  // const AccessLoc_t &getLoc() const {
  //   return loc;
  // }
  csi_id_t getAccID() const {
    if ((ver_acc_id & ID_MASK) == UNKNOWN_CSI_ACC_ID)
      return UNKNOWN_CSI_ID;
    return (ver_acc_id & ID_MASK);
  }
  uint16_t getVersion() const {
    return static_cast<uint16_t>(ver_acc_id >> VERSION_SHIFT);
  }
  AccessLoc_t getLoc() const {
    if (!isValid())
      return AccessLoc_t();
    return AccessLoc_t(getAccID(), *func->get_node()->get_call_stack());
  }

  MemoryAccess_t &operator=(const MemoryAccess_t &copy) {
    if (func != copy.func) {
      if (copy.func)
        copy.func->inc_ref_count();
      if (func)
        func->dec_ref_count();
      func = copy.func;
    }
    // loc = copy.loc;
    ver_acc_id = copy.ver_acc_id;

    // if (loc != copy.loc) {
    //   if (copy.loc)
    //     copy.loc->inc_ref_count();
    //   if (loc)
    //     loc->dec_ref_count();
    //   loc = copy.loc;
    // }
    return *this;
  }

  MemoryAccess_t &operator=(MemoryAccess_t &&move) {
    if (func)
      func->dec_ref_count();
    // if (loc)
    //   loc->dec_ref_count();
    func = move.func;
    // loc = std::move(move.loc);
    ver_acc_id = move.ver_acc_id;
    return *this;
  }

  void inc_ref_counts(int64_t count) {
    assert(func);
    // assert(loc);
    func->inc_ref_count(count);
    // loc.inc_ref_count(count);

    // loc->inc_ref_count(count);
  }

  void dec_ref_counts(int64_t count) {
    if (!func->dec_ref_count(count))
      func = nullptr;
    // loc.dec_ref_count(count);

    // if (!loc->dec_ref_count(count))
    //   loc = nullptr;
  }

  void inherit(const MemoryAccess_t &copy) {
    if (func)
      func->dec_ref_count();
    func = copy.func;
    // loc.dec_ref_count();

    // if (loc)
    //   loc->dec_ref_count();
    ver_acc_id = copy.ver_acc_id;
    // loc = copy.loc;
  }

  // // Unsafe method!  Only use this if you know what you're doing.
  // void overwrite(const MemoryAccess_t &copy) {
  //   func = copy.func;
  //   loc.overwrite(copy.loc);
  // }

  // // Unsafe method!  Only use this if you know what you're doing.
  // void clear() {
  //   func = nullptr;
  //   loc.clear();
  // }

  // bool sameAccessLocPtr(const MemoryAccess_t &that) const {
  //   return loc.getCallStack() == that.loc.getCallStack();
  // }

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
      // ", " << acc.loc;
      ", acc id " << acc.getAccID() << ", version " << acc.getVersion();
    return os;
  }

  // // Simple free-list allocator to conserve space and time in managing
  // // arrays of PAGE_SIZE MemoryAccess_t objects.
  // struct FreeNode_t {
  //   // static size_t FreeNode_ObjSize;
  //   FreeNode_t *next;
  // };
  // // TODO: Generalize this.
  // static const unsigned numFreeLists = 6;
  // static FreeNode_t *free_list[numFreeLists];

  // void *operator new[](size_t size) {
  //   unsigned lgSize = __builtin_ctzl(size);
  //   // if (!FreeNode_t::FreeNode_ObjSize)
  //   //   FreeNode_t::FreeNode_ObjSize = size;
  //   if (free_list[lgSize]) {
  //     // assert(size == FreeNode_t::FreeNode_ObjSize);
  //     FreeNode_t *new_node = free_list[lgSize];
  //     free_list[lgSize] = free_list[lgSize]->next;
  //     return new_node;
  //   }
  //   // std::cerr << "MemoryAccess_t::new[] called, size " << size << "\n";
  //   return ::operator new[](size);
  // }

  // void operator delete[](void *ptr) {
  //   FreeNode_t *del_node = reinterpret_cast<FreeNode_t *>(ptr);
  //   del_node->next = free_list;
  //   free_list = del_node;
  // }

  // static void cleanup_freelist() {
  //   for (unsigned i = 0; i < numFreeLists; ++i) {
  //     FreeNode_t *node = free_list[i];
  //     FreeNode_t *next = nullptr;
  //     while (node) {
  //       next = node->next;
  //       ::operator delete[](node);
  //       node = next;
  //     }
  //   }
  // }
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
