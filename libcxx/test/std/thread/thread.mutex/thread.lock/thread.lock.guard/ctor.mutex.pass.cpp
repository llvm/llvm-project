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
#include <type_traits>

#include "types.h"

int main(int, char**) {
  MyMutex m;
  assert(!m.locked);
  std::lock_guard<MyMutex> lg(m);
  assert(m.locked);

  static_assert(!std::is_convertible<MyMutex, std::lock_guard<MyMutex> >::value, "constructor must be explicit");

  return 0;
}
