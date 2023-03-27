//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: availability-shared_mutex-missing

// <shared_mutex>

// class shared_mutex;

// shared_mutex(const shared_mutex&) = delete;

#include <shared_mutex>

int main(int, char**)
{
    std::shared_mutex m0;
    std::shared_mutex m1(m0); // expected-error {{call to deleted constructor of 'std::shared_mutex'}}

  return 0;
}
