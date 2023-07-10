//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// stop_source();

#include <cassert>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_default_constructible_v<std::stop_source>);

int main(int, char**) {
  {
    std::stop_source ss = {}; // implicit
    assert(ss.stop_possible());
    assert(!ss.stop_requested());
  }

  return 0;
}
