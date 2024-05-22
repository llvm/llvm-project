//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <initializer_list>

// template<class E> constexpr const E* begin(initializer_list<E> il) noexcept; // constexpr since C++14
// template<class E> constexpr const E* end(initializer_list<E> il) noexcept;   // constexpr since C++14

#include <initializer_list>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

TEST_CONSTEXPR_CXX14 bool test() {
  // unqualified begin/end
  {
    std::initializer_list<int> il = {3, 2, 1};
    ASSERT_NOEXCEPT(begin(il));
    ASSERT_NOEXCEPT(end(il));
    ASSERT_SAME_TYPE(decltype(begin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(end(il)), const int*);
    const int* b = begin(il);
    const int* e = end(il);
    assert(il.size() == 3);
    assert(static_cast<std::size_t>(e - b) == il.size());
    assert(*b++ == 3);
    assert(*b++ == 2);
    assert(*b++ == 1);
  }

  // qualified begin/end
  {
    std::initializer_list<int> il = {1, 2, 3};
    ASSERT_NOEXCEPT(std::begin(il));
    ASSERT_NOEXCEPT(std::end(il));
    ASSERT_SAME_TYPE(decltype(std::begin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(il)), const int*);
    assert(std::begin(il) == il.begin());
    assert(std::end(il) == il.end());

    const auto& cil = il;
    ASSERT_NOEXCEPT(std::begin(cil));
    ASSERT_NOEXCEPT(std::end(cil));
    ASSERT_SAME_TYPE(decltype(std::begin(cil)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(cil)), const int*);
    assert(std::begin(cil) == il.begin());
    assert(std::end(cil) == il.end());
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif

  return 0;
}
