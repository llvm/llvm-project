//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// void swap(path& lhs, path& rhs) noexcept;

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "count_new.hpp"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;

// NOTE: this is tested in path.members/path.modifiers via the member swap.
int main()
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
}
