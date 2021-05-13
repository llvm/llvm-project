//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11, c++14

// shared_mutex was introduced in macosx10.12
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.11
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.10
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.9

// <shared_mutex>

// class shared_mutex;

// shared_mutex();

#include <shared_mutex>

#include "test_macros.h"

int main(int, char**)
{
    std::shared_mutex m;

  return 0;
}
