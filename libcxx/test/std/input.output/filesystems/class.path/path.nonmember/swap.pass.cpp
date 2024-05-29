//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// void swap(path& lhs, path& rhs) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"
namespace fs = std::filesystem;

// NOTE: this is tested in path.members/path.modifiers via the member swap.
int main(int, char**)
{
  using namespace fs;
  const char* value1 = "foo/bar/baz";
  const char* value2 = "_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG";
  {
    path p1(value1);
    path p2(value2);
    fs::path::string_type ps1 = p1.native();
    fs::path::string_type ps2 = p2.native();

    DisableAllocationGuard g;
    swap(p1, p2);
    ASSERT_NOEXCEPT(swap(p1, p2));
    ASSERT_SAME_TYPE(void, decltype(swap(p1, p2)));
    assert(p1.native() == ps2);
    assert(p2.native() == ps1);
    swap(p1, p2);
    assert(p1.native() == ps1);
    assert(p2.native() == ps2);
  }

  // Test the swap two-step idiom
  {
    path p1(value1);
    path p2(value2);
    fs::path::string_type ps1 = p1.native();
    fs::path::string_type ps2 = p2.native();

    DisableAllocationGuard g;
    using namespace std;
    swap(p1, p2);
    ASSERT_NOEXCEPT(swap(p1, p2));
    ASSERT_SAME_TYPE(void, decltype(swap(p1, p2)));
    assert(p1.native() == ps2);
    assert(p2.native() == ps1);
    swap(p1, p2);
    assert(p1.native() == ps1);
    assert(p2.native() == ps2);
  }

  return 0;
}
