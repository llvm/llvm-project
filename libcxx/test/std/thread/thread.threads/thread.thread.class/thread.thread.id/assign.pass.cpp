//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <thread>

// class thread::id

// id& operator=(const id&) = default;

#include <thread>
#include <cassert>

int main(int, char**)
{
    std::thread::id id0;
    std::thread::id id1;
    id1 = id0;
    assert(id1 == id0);
    id1 = std::this_thread::get_id();
    assert(id1 != id0);

  return 0;
}
