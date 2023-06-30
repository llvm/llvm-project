//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class path

// void swap(path& rhs) noexcept;

#include "filesystem_include.h"
#include <cassert>
#include <type_traits>

#include "assert_macros.h"
#include "count_new.h"
#include "test_iterators.h"


struct SwapTestcase {
  const char* value1;
  const char* value2;
};

#define LONG_STR1 "_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG_THIS_IS_LONG"
#define LONG_STR2 "_THIS_IS_LONG2_THIS_IS_LONG2_THIS_IS_LONG2_THIS_IS_LONG2_THIS_IS_LONG2_THIS_IS_LONG2_THIS_IS_LONG2"
const SwapTestcase TestCases[] =
  {
      {"", ""}
    , {"shortstr", LONG_STR1}
    , {LONG_STR1, "shortstr"}
    , {LONG_STR1, LONG_STR2}
  };
#undef LONG_STR1
#undef LONG_STR2

int main(int, char**)
{
  using namespace fs;
  {
    path p;
    ASSERT_NOEXCEPT(p.swap(p));
    ASSERT_SAME_TYPE(void, decltype(p.swap(p)));
  }
  for (auto const & TC : TestCases) {
    path p1(TC.value1);
    path p2(TC.value2);
    {
      DisableAllocationGuard g;
      p1.swap(p2);
    }
    assert(p1 == TC.value2);
    assert(p2 == TC.value1);
    {
      DisableAllocationGuard g;
      p1.swap(p2);
    }
    assert(p1 == TC.value1);
    assert(p2 == TC.value2);
  }
  // self-swap
  {
    const char* Val = "aoeuaoeuaoeuaoeuaoeuaoeuaoeuaoeuaoeu";
    path p1(Val);
    assert(p1 == Val);
    {
      DisableAllocationGuard g;
      p1.swap(p1);
    }
    assert(p1 == Val);
  }

  return 0;
}
