//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <type_traits>

// constexpr bool is_constant_evaluated() noexcept; // C++20

#include <type_traits>
#include <cassert>

#include "test_macros.h"

#ifndef __cpp_lib_is_constant_evaluated
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
# error __cpp_lib_is_constant_evaluated should be defined
#endif
#endif

template <bool> struct InTemplate {};

int main(int, char**)
{
#ifdef __cpp_lib_is_constant_evaluated
  // Test the signature
  {
    ASSERT_SAME_TYPE(decltype(std::is_constant_evaluated()), bool);
    ASSERT_NOEXCEPT(std::is_constant_evaluated());
    constexpr bool p = std::is_constant_evaluated();
    assert(p);
  }
  // Test the return value of the builtin for basic sanity only. It's the
  // compilers job to test tho builtin for correctness.
  {
    static_assert(std::is_constant_evaluated(), "");
    bool p = std::is_constant_evaluated();
    assert(!p);
    ASSERT_SAME_TYPE(InTemplate<std::is_constant_evaluated()>, InTemplate<true>);
    static int local_static = std::is_constant_evaluated() ? 42 : -1;
    assert(local_static == 42);
  }
#endif
  return 0;
}
