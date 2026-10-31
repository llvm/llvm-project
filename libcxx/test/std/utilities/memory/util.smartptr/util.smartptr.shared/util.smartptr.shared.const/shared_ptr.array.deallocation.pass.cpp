//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: sanitizer-new-delete

// <memory>

// template<class Y> explicit shared_ptr(Y* p);

#include <memory>
#include <new>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

// virtual destructor forces the compiler to emit an array cookie
struct ElemVirtualDtor {
  virtual ~ElemVirtualDtor() {}
};

// non-trivial destructor forces the compiler to emit an array cookie
struct ElemNonTrivialDtor {
  int v = 0;
  ~ElemNonTrivialDtor() { v = 42; }
};

template <typename T, std::size_t N>
void run() {
  globalMemCounter.reset();
#if TEST_STD_VER >= 17
  T* arr = new T[N];
  assert(globalMemCounter.checkOutstandingArrayNewEq(1));
  globalMemCounter.throw_after = 0;

  try {
    std::shared_ptr<T[]> sp(arr);
    assert(false);
  } catch (const std::bad_alloc&) {
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(globalMemCounter.outstanding_array_new == 0);
  }
#else
  T* p                         = new T;
  globalMemCounter.throw_after = 0;
  try {
    std::shared_ptr<T> sp(p);
    assert(false);
  } catch (const std::bad_alloc&) {
    assert(globalMemCounter.checkOutstandingNewEq(0));
  }
#endif
}

int main(int, char**) {
  const std::size_t N = 10;
  run<ElemVirtualDtor, N>();
  run<ElemNonTrivialDtor, N>();
  return 0;
}
