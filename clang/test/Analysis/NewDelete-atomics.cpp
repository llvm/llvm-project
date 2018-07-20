// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDeleteLeaks -DLEAKS -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDeleteLeaks -DLEAKS -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s
// RUN: %clang_analyze_cc1 -analyzer-inline-max-stack-depth 2 -analyzer-config ipa-always-inline-size=2 -analyzer-checker=core,cplusplus.NewDelete -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-inline-max-stack-depth 2 -analyzer-config ipa-always-inline-size=2 -analyzer-checker=core,cplusplus.NewDeleteLeaks -DLEAKS -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-inline-max-stack-depth 2 -analyzer-config ipa-always-inline-size=2 -analyzer-checker=core,cplusplus.NewDelete -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s
// RUN: %clang_analyze_cc1 -analyzer-inline-max-stack-depth 2 -analyzer-config ipa-always-inline-size=2 -analyzer-checker=core,cplusplus.NewDeleteLeaks -DLEAKS -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s

// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_consume = __ATOMIC_CONSUME,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

class Obj {
  int RefCnt;

public:
  int incRef() {
    return __c11_atomic_fetch_add((volatile _Atomic(int) *)&RefCnt, 1,
                                  memory_order_relaxed);
  }

  int decRef() {
    return __c11_atomic_fetch_sub((volatile _Atomic(int) *)&RefCnt, 1,
                                  memory_order_relaxed);
  }

  void foo();
};

class IntrusivePtr {
  Obj *Ptr;

public:
  IntrusivePtr(Obj *Ptr) : Ptr(Ptr) {
    Ptr->incRef();
  }

  IntrusivePtr(const IntrusivePtr &Other) : Ptr(Other.Ptr) {
    Ptr->incRef();
  }

  ~IntrusivePtr() {
  // We should not take the path on which the object is deleted.
    if (Ptr->decRef() == 1)
      delete Ptr;
  }

  Obj *getPtr() const { return Ptr; } // no-warning
};

void testDestroyLocalRefPtr() {
  IntrusivePtr p1(new Obj());
  {
    IntrusivePtr p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}

void testDestroySymbolicRefPtr(const IntrusivePtr &p1) {
  {
    IntrusivePtr p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}
