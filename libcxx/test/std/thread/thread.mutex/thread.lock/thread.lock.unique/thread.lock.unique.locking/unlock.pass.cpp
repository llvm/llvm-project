//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// void unlock();

#include <cassert>
#include <mutex>
#include <system_error>

#include "test_macros.h"
#include "../types.h"

MyMutex m;

int main(int, char**) {
  std::unique_lock<MyMutex> lk(m);
  lk.unlock();
  assert(lk.owns_lock() == false);
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    lk.unlock();
    assert(false);
  } catch (std::system_error& e) {
    assert(e.code() == std::errc::operation_not_permitted);
  }
#endif
  lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    lk.unlock();
    assert(false);
  } catch (std::system_error& e) {
    assert(e.code() == std::errc::operation_not_permitted);
  }
#endif

  return 0;
}
