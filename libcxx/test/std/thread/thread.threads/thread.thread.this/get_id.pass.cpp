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

// thread::id this_thread::get_id();

#include <thread>
#include <cassert>

int main()
{
    std::thread::id id = std::this_thread::get_id();
    assert(id != std::thread::id());
}
