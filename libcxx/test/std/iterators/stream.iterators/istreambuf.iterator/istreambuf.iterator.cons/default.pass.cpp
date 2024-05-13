//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator
//
// istreambuf_iterator() noexcept; // constexpr since C++11
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  {
    typedef std::istreambuf_iterator<char> T;
    T it;
    assert(it == T());
#if TEST_STD_VER >= 11
    constexpr T it2;
    (void)it2;
#endif

    ASSERT_NOEXCEPT(T());
  }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    typedef std::istreambuf_iterator<wchar_t> T;
    T it;
    assert(it == T());
#if TEST_STD_VER >= 11
    constexpr T it2;
    (void)it2;
#endif

    ASSERT_NOEXCEPT(T());
  }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
