//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock();

#include <mutex>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::unique_lock<std::mutex> ul;
    assert(!ul.owns_lock());
    assert(ul.mutex() == nullptr);

  return 0;
}
