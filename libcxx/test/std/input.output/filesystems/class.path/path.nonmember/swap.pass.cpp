//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// void swap(path& lhs, path& rhs) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


// NOTE: this is tested in path.members/path.modifiers via the member swap.
int main(int, char**)
{
  using namespace fs;
  const char* value1 = "foo/bar/baz";
  const char* value2 = "_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG";
  path p1(value1);
  path p2(value2);
  {
    using namespace std; using namespace fs;
    ASSERT_NOEXCEPT(swap(p1, p2));
    ASSERT_SAME_TYPE(void, decltype(swap(p1, p2)));
  }
  {
    DisableAllocationGuard g;
    using namespace std;
    using namespace fs;
    swap(p1, p2);
    assert(p1.native() == value2);
    assert(p2.native() == value1);
    swap(p1, p2);
    assert(p1.native() == value1);
    assert(p2.native() == value2);
  }

  return 0;
}
