//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: windows

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0

// This test requires debug mode, which the library on macOS doesn't have.
// UNSUPPORTED: with_system_cxx_lib=macosx

// <filesystem>

// class path

#include "filesystem_include.h"
#include <iterator>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "debug_mode_helper.h"

int main(int, char**) {
  using namespace fs;
  // Test incrementing/decrementing a singular iterator
  {
    path::iterator singular;
    EXPECT_DEATH( ++singular );
    EXPECT_DEATH( --singular );
  }
  // Test decrementing the begin iterator
  {
    path p("foo/bar");
    auto it = p.begin();
    ++it;
    ++it;
    EXPECT_DEATH( ++it );
  }
  // Test incrementing the end iterator
  {
    path p("foo/bar");
    auto it = p.end();
    EXPECT_DEATH( ++it );
    --it;
    --it;
    EXPECT_DEATH( --it );
  }

  return 0;
}
