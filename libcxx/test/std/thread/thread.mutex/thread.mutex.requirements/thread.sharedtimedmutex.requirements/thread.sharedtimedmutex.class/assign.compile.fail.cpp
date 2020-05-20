//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// shared_timed_mutex was introduced in macosx10.12
// UNSUPPORTED: with_system_cxx_lib=macosx10.11
// UNSUPPORTED: with_system_cxx_lib=macosx10.10
// UNSUPPORTED: with_system_cxx_lib=macosx10.9

// <shared_mutex>

// class shared_timed_mutex;

// shared_timed_mutex& operator=(const shared_timed_mutex&) = delete;

#include <shared_mutex>

int main(int, char**)
{
    std::shared_timed_mutex m0;
    std::shared_timed_mutex m1;
    m1 = m0;

  return 0;
}
