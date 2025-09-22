//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// istream_iterator(const istream_iterator& x) noexcept(see below);

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

// The copy constructor is constexpr in C++11, but that is not easy to test.
// The comparison of the class is not constexpr so this is only a compile test.
TEST_CONSTEXPR_CXX14 bool test_constexpr() {
  std::istream_iterator<int> io;
  [[maybe_unused]] std::istream_iterator<int> i = io;

  return true;
}

struct throwing_copy_constructor {
  throwing_copy_constructor() {}
  throwing_copy_constructor(const throwing_copy_constructor&) TEST_NOEXCEPT_FALSE {}
};

int main(int, char**)
{
    {
        std::istream_iterator<int> io;
        std::istream_iterator<int> i = io;
        assert(i == std::istream_iterator<int>());
#if TEST_STD_VER >= 11
        static_assert(std::is_nothrow_copy_constructible<std::istream_iterator<int>>::value, "");
#endif
    }
    {
      std::istream_iterator<throwing_copy_constructor> io;
      std::istream_iterator<throwing_copy_constructor> i = io;
      assert(i == std::istream_iterator<throwing_copy_constructor>());
#if TEST_STD_VER >= 11
      static_assert(!std::is_nothrow_copy_constructible<std::istream_iterator<throwing_copy_constructor>>::value, "");
#endif
    }
    {
        std::istringstream inf(" 1 23");
        std::istream_iterator<int> io(inf);
        std::istream_iterator<int> i = io;
        assert(i != std::istream_iterator<int>());
        int j = 0;
        j = *i;
        assert(j == 1);
    }

#if TEST_STD_VER >= 14
    static_assert(test_constexpr(), "");
#endif

    return 0;
}
