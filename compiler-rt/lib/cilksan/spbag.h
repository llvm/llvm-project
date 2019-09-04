/* -*- Mode: C++ -*- */

#ifndef _SPBAG_H
#define _SPBAG_H

#include <assert.h>
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <inttypes.h>

#include "debug_util.h"
#include "race_info.h"

#define UNINIT_STACK_PTR ((uintptr_t)0LL)

class SPBagInterface {
public:
  // Note to self: base class must declare a virtual destructor; it does not
  // have to be pure and must provide a definition.
  // http://stackoverflow.com/questions/461203/when-to-use-virtual-destructors
  // <stackoverflow>/3336499/virtual-desctructor-on-pure-abstract-base-class
  virtual ~SPBagInterface() { }
  virtual bool is_SBag() const = 0;
  virtual bool is_PBag() const = 0;
  virtual uint64_t get_func_id() const = 0;
  virtual uint64_t get_rsp() const = 0;
  virtual void set_rsp(uint64_t stack_ptr) = 0;
  virtual uint16_t get_version() const = 0;
  virtual bool inc_version() = 0;
  // virtual std::string get_call_context() = 0;
  virtual const call_stack_t *get_call_stack() const = 0;
  virtual void update_sibling(SPBagInterface *) = 0;
};


class SBag_t : public SPBagInterface {
private:
  uint64_t _func_id;
  // std::string _func_name;
  // SBag of the parent function (whether this function is called or spawned)
  // SPBagInterface *_parent;
  static constexpr unsigned VERSION_SHIFT = 48;
  static constexpr uintptr_t STACK_PTR_MASK = ((1UL << VERSION_SHIFT) - 1);
  static constexpr uintptr_t VERSION_MASK = ~STACK_PTR_MASK;
  uintptr_t _ver_stack_ptr;
  call_stack_t _call_stack;

  SBag_t() = delete;  // disable default constructor

public:
  SBag_t(uint64_t id, const call_stack_t &call_stack) :
      _func_id(id),
      // _func_name(name),
      // _parent(parent),
      // _stack_ptr(UNINIT_STACK_PTR),
      _ver_stack_ptr(UNINIT_STACK_PTR), _call_stack(call_stack) {
    // std::cerr << "  Constructing SBag_t " << (void*)this << "\n";
    WHEN_CILKSAN_DEBUG(debug_count++);
  }

#if CILKSAN_DEBUG
  static long debug_count;
  ~SBag_t() {
    // std::cerr << "  Destructing SBag_t " << (void*)this << "\n";
    debug_count--;
  }
#endif

  bool is_SBag() const { return true; }
  bool is_PBag() const { return false; }

  uint64_t get_func_id() const { return _func_id; }

  uint64_t get_rsp() const {
    cilksan_assert((_ver_stack_ptr & STACK_PTR_MASK) != UNINIT_STACK_PTR);
    return (_ver_stack_ptr & STACK_PTR_MASK);
  }
  void set_rsp(uintptr_t stack_ptr) {
    // _stack_ptr = stack_ptr;
    _ver_stack_ptr = (stack_ptr & STACK_PTR_MASK) |
      (_ver_stack_ptr & VERSION_MASK);
  }

  uint16_t get_version() const {
    // return _version;
    return (_ver_stack_ptr >> VERSION_SHIFT);
  }
  bool inc_version() {
    // std::cerr << "inc_version(): was version = " << _version << "\n";
    // return (0 != ++_version);
    uintptr_t new_version =
      (_ver_stack_ptr & VERSION_MASK) + (1UL << VERSION_SHIFT);
    _ver_stack_ptr = (_ver_stack_ptr & STACK_PTR_MASK) | new_version;
    return (0 != new_version);
  }

  const call_stack_t *get_call_stack() const {
    return &_call_stack;
  }

  // Note to self: Apparently the compiler will generate a default inline
  // destructor, and it's better to let the compiler to that than define your
  // own empty destructor.  This is true even when the parent class has a
  // virtual destructor.
  // http://stackoverflow.com/questions/827196/virtual-default-destructors-in-c
  // ~SBag_t() { fprintf(stderr, "Called SBag destructor.\n"); }

  /*
  std::string get_call_context() {
    std::string res;
    if(_parent) {
      res = _func_name + " called in \n" + _parent->get_call_context();
    } else {
      res = _func_name + "\n";
    }
    return res;
  }*/

  void update_sibling(SPBagInterface *) {
    cilksan_assert(0 && "update_sibling called from SBag_t");
  }

  // Simple free-list allocator to conserve space and time in managing
  // SBag_t objects.

  // The structure of a node in the SBag free list.
  struct FreeNode_t {
    FreeNode_t *next = nullptr;
  };
  static FreeNode_t *free_list;

  void *operator new(size_t size) {
    if (free_list) {
      FreeNode_t *new_node = free_list;
      free_list = free_list->next;
      return new_node;
    }
    return ::operator new(size);
  }

  void operator delete(void *ptr) {
    FreeNode_t *del_node = reinterpret_cast<FreeNode_t *>(ptr);
    del_node->next = free_list;
    free_list = del_node;
  }

  static void cleanup_freelist() {
    FreeNode_t *node = free_list;
    FreeNode_t *next = nullptr;
    while (node) {
      next = node->next;
      ::operator delete(node);
      node = next;
    }
  }
};

static_assert(sizeof(SBag_t) >= sizeof(SBag_t::FreeNode_t),
              "Node structure in SBag free list must be as large as SBag.");

class PBag_t : public SPBagInterface {
private:
  // the SBag that corresponds to the function instance that holds this PBag
  SPBagInterface *_sib_sbag = nullptr;

  PBag_t() = delete; // disable default constructor

public:
  PBag_t(SPBagInterface *sib) : _sib_sbag(sib) {
    // std::cerr << "  Constructing PBag_t " << (void*)this << "\n";
    WHEN_CILKSAN_DEBUG( debug_count++; );
  }

#if CILKSAN_DEBUG
  static long debug_count;
  ~PBag_t() {
    // std::cerr << "  Destructing PBag_t " << (void*)this << "\n";
    debug_count--;
    _sib_sbag = nullptr;
  }
#endif

  bool is_SBag() const { return false; }
  bool is_PBag() const { return true; }
  uint64_t get_func_id() const { return _sib_sbag->get_func_id(); }
  uint64_t get_rsp() const { return _sib_sbag->get_rsp(); }
  void set_rsp(uint64_t stack_ptr) {
     /* Should never happen; */
    cilksan_assert(0 && "Called set_rsp on a Pbag");
  }
  uint16_t get_version() const {
    cilksan_assert(0 && "Called get_version on a Pbag");
    return _sib_sbag->get_version();
  }
  bool inc_version() {
    cilksan_assert(0 && "Called inc_version on a Pbag");
    return _sib_sbag->inc_version();
  }
  const call_stack_t *get_call_stack() const {
     /* Should never happen; */
    cilksan_assert(0 && "Called get_call_stack on a Pbag");
    return _sib_sbag->get_call_stack();
  }

  /*
  std::string get_call_context() {
    return _sib_sbag->get_call_context();
  } */

  void update_sibling(SPBagInterface *new_sib) {
    _sib_sbag = new_sib;
  }

  // Simple free-list allocator to conserve space and time in managing
  // PBag_t objects.
  struct FreeNode_t {
    FreeNode_t *next = nullptr;
  };
  static FreeNode_t *free_list;

  void *operator new(size_t size) {
    if (free_list) {
      FreeNode_t *new_node = free_list;
      free_list = free_list->next;
      return new_node;
    }
    return ::operator new(size);
  }

  void operator delete(void *ptr) {
    FreeNode_t *del_node = reinterpret_cast<FreeNode_t *>(ptr);
    del_node->next = free_list;
    free_list = del_node;
  }

  static void cleanup_freelist() {
    FreeNode_t *node = free_list;
    FreeNode_t *next = nullptr;
    while (node) {
      next = node->next;
      ::operator delete(node);
      node = next;
    }
  }
};

static_assert(sizeof(PBag_t) >= sizeof(PBag_t::FreeNode_t),
              "Node structure in PBag free list must be as large as PBag.");

#endif // #ifndef _SPBAG_H
