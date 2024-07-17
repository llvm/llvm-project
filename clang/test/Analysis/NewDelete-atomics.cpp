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

class RawObj {
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

class StdAtomicObj {
  std::atomic<int> RefCnt;

public:
  int incRef() {
    return ++RefCnt;
  }

  int decRef() {
    return --RefCnt;
  }

  void foo();
};

template <typename T>
class IntrusivePtr {
  T *Ptr;

public:
  IntrusivePtr(T *Ptr) : Ptr(Ptr) {
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

  T *getPtr() const { return Ptr; } // no-warning
};

// Also IntrusivePtr but let's dodge name-based heuristics.
template <typename T>
class DifferentlyNamed {
  T *Ptr;

public:
  DifferentlyNamed(T *Ptr) : Ptr(Ptr) {
    Ptr->incRef();
  }

  DifferentlyNamed(const DifferentlyNamed &Other) : Ptr(Other.Ptr) {
    Ptr->incRef();
  }

  ~DifferentlyNamed() {
  // We should not take the path on which the object is deleted.
    if (Ptr->decRef() == 1)
      delete Ptr;
  }

  T *getPtr() const { return Ptr; } // no-warning
};

void testDestroyLocalRefPtr() {
  IntrusivePtr<RawObj> p1(new RawObj());
  {
    IntrusivePtr<RawObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}

void testDestroySymbolicRefPtr(const IntrusivePtr<RawObj> &p1) {
  {
    IntrusivePtr<RawObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}

void testDestroyLocalRefPtrWithAtomics() {
  IntrusivePtr<StdAtomicObj> p1(new StdAtomicObj());
  {
    IntrusivePtr<StdAtomicObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}


void testDestroyLocalRefPtrWithAtomics(const IntrusivePtr<StdAtomicObj> &p1) {
  {
    IntrusivePtr<StdAtomicObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}

void testDestroyLocalRefPtrDifferentlyNamed() {
  DifferentlyNamed<RawObj> p1(new RawObj());
  {
    DifferentlyNamed<RawObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}

void testDestroySymbolicRefPtrDifferentlyNamed(
    const DifferentlyNamed<RawObj> &p1) {
  {
    DifferentlyNamed<RawObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}

void testDestroyLocalRefPtrWithAtomicsDifferentlyNamed() {
  DifferentlyNamed<StdAtomicObj> p1(new StdAtomicObj());
  {
    DifferentlyNamed<StdAtomicObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}


void testDestroyLocalRefPtrWithAtomicsDifferentlyNamed(
    const DifferentlyNamed<StdAtomicObj> &p1) {
  {
    DifferentlyNamed<StdAtomicObj> p2(p1);
  }

  // p1 still maintains ownership. The object is not deleted.
  p1.getPtr()->foo(); // no-warning
}
