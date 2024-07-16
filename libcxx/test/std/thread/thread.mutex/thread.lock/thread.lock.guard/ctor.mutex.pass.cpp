//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

#include <cassert>
#include <mutex>

#include "types.h"

int main(int, char**) {
  MyMutex m;
  std::lock_guard<MyMutex> lg(m); // makes sure this compiles and runs
  static_assert(!std::is_convertible_v<MyMutex, std::lock_guard<MyMutex>>, "constructor must be explicit");

  return 0;
}
