//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// ~basic_string() // implied noexcept; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class T>
struct throwing_alloc
{
    typedef T value_type;
    throwing_alloc(const throwing_alloc&);
    T *allocate(size_t);
    ~throwing_alloc() noexcept(false);
};

// Test that it's possible to take the address of basic_string's destructors
// by creating globals which will register their destructors with cxa_atexit.
std::string unused_string;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
std::wstring unused_wide_string;
#endif

static_assert(std::is_nothrow_destructible<std::string>::value, "");
static_assert(std::is_nothrow_destructible<
                std::basic_string<char, std::char_traits<char>, test_allocator<char>>>::value, "");
LIBCPP_STATIC_ASSERT(!std::is_nothrow_destructible<
                     std::basic_string<char, std::char_traits<char>, throwing_alloc<char>>>::value, "");

TEST_CONSTEXPR_CXX20 bool test() {
  test_allocator_statistics alloc_stats;
  {
    std::basic_string<char, std::char_traits<char>, test_allocator<char>> str2((test_allocator<char>(&alloc_stats)));
    str2 = "long long string so no SSO";
    assert(alloc_stats.alloc_count > 0);
    LIBCPP_ASSERT(alloc_stats.alloc_count == 1);
  }
  assert(alloc_stats.alloc_count == 0);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
