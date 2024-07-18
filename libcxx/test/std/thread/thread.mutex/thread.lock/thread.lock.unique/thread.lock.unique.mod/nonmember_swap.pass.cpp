//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// template <class Mutex>
//   void swap(unique_lock<Mutex>& x, unique_lock<Mutex>& y);

#include <cassert>
#include <mutex>

#include "test_macros.h"
#include "../types.h"

MyMutex m;

int main(int, char**) {
  std::unique_lock<MyMutex> lk1(m);
  std::unique_lock<MyMutex> lk2;
  swap(lk1, lk2);
  assert(lk1.mutex() == nullptr);
  assert(lk1.owns_lock() == false);
  assert(lk2.mutex() == &m);
  assert(lk2.owns_lock() == true);

  return 0;
}
