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
  static constexpr unsigned TYPE_SHIFT = 44;
  static constexpr csi_id_t ID_MASK = ((1UL << TYPE_SHIFT) - 1);
  static constexpr csi_id_t TYPE_MASK = ((1UL << VERSION_SHIFT) - 1) & ~ID_MASK;
  static constexpr csi_id_t UNKNOWN_CSI_ACC_ID = UNKNOWN_CSI_ID & ID_MASK;
public:
  DisjointSet_t<SPBagInterface *> *func = nullptr;
  csi_id_t ver_acc_id = UNKNOWN_CSI_ACC_ID;

  MemoryAccess_t() {}
  MemoryAccess_t(DisjointSet_t<SPBagInterface *> *func, csi_id_t acc_id,
                 MAType_t type)
      : func(func), ver_acc_id((acc_id & ID_MASK) |
                               (static_cast<csi_id_t>(type) << TYPE_SHIFT))
  {
    if (func) {
      func->inc_ref_count();
      ver_acc_id |=
        static_cast<csi_id_t>(func->get_node()->get_version()) << VERSION_SHIFT;
    }
  }

  MemoryAccess_t(const MemoryAccess_t &copy)
      : func(copy.func), ver_acc_id(copy.ver_acc_id) {
    if (func)
      func->inc_ref_count();
  }

  MemoryAccess_t(const MemoryAccess_t &&move)
      : func(move.func), ver_acc_id(move.ver_acc_id) {}

  ~MemoryAccess_t() {
    if (func) {
      func->dec_ref_count();
      func = nullptr;
    }
  }

  bool isValid() const {
    return (nullptr != func);
  }

  void invalidate() {
    if (func)
      func->dec_ref_count();
    func = nullptr;
    ver_acc_id = UNKNOWN_CSI_ACC_ID;
  }

  DisjointSet_t<SPBagInterface *> *getFunc() const {
    return func;
  }

  csi_id_t getAccID() const {
    if ((ver_acc_id & ID_MASK) == UNKNOWN_CSI_ACC_ID)
      return UNKNOWN_CSI_ID;
    return (ver_acc_id & ID_MASK);
  }
  MAType_t getAccType() const {
    if ((ver_acc_id & ID_MASK) == UNKNOWN_CSI_ACC_ID)
      return MAType_t::UNKNOWN;
    return static_cast<MAType_t>((ver_acc_id & TYPE_MASK) >> TYPE_SHIFT);
  }
  uint16_t getVersion() const {
    return static_cast<uint16_t>(ver_acc_id >> VERSION_SHIFT);
  }
  AccessLoc_t getLoc() const {
    if (!isValid())
      return AccessLoc_t();
    return AccessLoc_t(getAccID(), getAccType(),
                       *func->get_node()->get_call_stack());
  }

  MemoryAccess_t &operator=(const MemoryAccess_t &copy) {
    if (func != copy.func) {
      if (copy.func)
        copy.func->inc_ref_count();
      if (func)
        func->dec_ref_count();
      func = copy.func;
    }
    ver_acc_id = copy.ver_acc_id;

    return *this;
  }

  MemoryAccess_t &operator=(MemoryAccess_t &&move) {
    if (func)
      func->dec_ref_count();
    func = move.func;
    ver_acc_id = move.ver_acc_id;
    return *this;
  }

  void inc_ref_counts(int64_t count) {
    assert(func);
    func->inc_ref_count(count);
  }

  void dec_ref_counts(int64_t count) {
    if (!func->dec_ref_count(count))
      func = nullptr;
  }

  void inherit(const MemoryAccess_t &copy) {
    if (func)
      func->dec_ref_count();
    func = copy.func;
    ver_acc_id = copy.ver_acc_id;
  }

  // TODO: Get rid of PC from these comparisons
  bool operator==(const MemoryAccess_t &that) const {
    return (func == that.func);
  }

  bool operator!=(const MemoryAccess_t &that) const {
    return !(*this == that);
  }

  inline friend
  std::ostream& operator<<(std::ostream &os, const MemoryAccess_t &acc) {
    os << "function " << acc.func->get_node()->get_func_id()
       << ", acc id " << acc.getAccID() << ", type " << acc.getAccType()
       << ", version " << acc.getVersion();
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
