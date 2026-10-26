//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++1, c++14
// UNSUPPORTED: no-exceptions

// <memory>

// template<class Y> explicit shared_ptr(Y* p);

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

// virtual destructor forces the compiler to emit an array cookie
struct ElemVirtualDtor {
  int v                      = 0;
  virtual ~ElemVirtualDtor() = default;
};

// non-trivial destructor forces the compiler to emit an array cookie
struct ElemNonTrivialDtor {
  int v = 0;
  ~ElemNonTrivialDtor() { v = 42; }
};

// custom `operator delete[](void* p, std::size_t s)` forces the compiler to emit an array cookie
struct ElemWithDeleter {
  int v = 0;
  // a class-specific sized operator delete[] also requires an array cookie.
  static void operator delete[](void* p, std::size_t s) noexcept { ::operator delete[](p, s); }
};

template <typename T, std::size_t N>
void run() {
  globalMemCounter.reset();
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
}

int main(int, char**) {
  constexpr std::size_t N = 10;
  run<ElemVirtualDtor, N>();
  run<ElemNonTrivialDtor, N>();
  run<ElemWithDeleter, N>();
  return 0;
}
