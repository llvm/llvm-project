//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// UNSUPPORTED: availability-shared_mutex-missing

// <shared_mutex>

// class shared_timed_mutex;

// shared_timed_mutex(const shared_timed_mutex&) = delete;

#include <shared_mutex>

int main(int, char**)
{
    std::shared_timed_mutex m0;
    std::shared_timed_mutex m1(m0);

  return 0;
}
