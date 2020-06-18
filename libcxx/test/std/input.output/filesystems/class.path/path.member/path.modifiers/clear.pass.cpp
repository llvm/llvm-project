//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// void clear() noexcept

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


int main(int, char**) {
  using namespace fs;
  {
    path p;
    ASSERT_NOEXCEPT(p.clear());
    ASSERT_SAME_TYPE(void, decltype(p.clear()));
    p.clear();
    assert(p.empty());
  }
  {
    const path p("/foo/bar/baz");
    path p2(p);
    assert(p == p2);
    p2.clear();
    assert(p2.empty());
  }

  return 0;
}
