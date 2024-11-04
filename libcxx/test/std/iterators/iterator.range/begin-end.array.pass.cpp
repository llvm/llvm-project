//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <iterator>
//
// template <class T, size_t N> constexpr T* begin(T (&array)[N]) noexcept;
// template <class T, size_t N> constexpr T* end(T (&array)[N]) noexcept;
//
// template <class T, size_t N> constexpr reverse_iterator<T*> rbegin(T (&array)[N]);      // C++14, constexpr since C++17
// template <class T, size_t N> constexpr reverse_iterator<T*> rend(T (&array)[N]);        // C++14, constexpr since C++17

#include <cassert>
#include <iterator>

#include "test_macros.h"

TEST_CONSTEXPR_CXX14 bool test() {
  int a[]        = {1, 2, 3};
  const auto& ca = a;

  // std::begin(T (&)[N]) / std::end(T (&)[N])
  {
    ASSERT_NOEXCEPT(std::begin(a));
    ASSERT_SAME_TYPE(decltype(std::begin(a)), int*);
    assert(std::begin(a) == a);

    ASSERT_NOEXCEPT(std::end(a));
    ASSERT_SAME_TYPE(decltype(std::end(a)), int*);
    assert(std::end(a) == a + 3);

    // kind of overkill since it follows from the definition, but worth testing
    ASSERT_NOEXCEPT(std::begin(ca));
    ASSERT_SAME_TYPE(decltype(std::begin(ca)), const int*);
    assert(std::begin(ca) == ca);

    ASSERT_NOEXCEPT(std::end(ca));
    ASSERT_SAME_TYPE(decltype(std::end(ca)), const int*);
    assert(std::end(ca) == ca + 3);
  }

  return true;
}

TEST_CONSTEXPR_CXX17 bool test_r() {
#if TEST_STD_VER >= 14
  int a[]        = {1, 2, 3};
  const auto& ca = a;

  // std::rbegin(T (&)[N]) / std::rend(T (&)[N])
  {
    ASSERT_SAME_TYPE(decltype(std::rbegin(a)), std::reverse_iterator<int*>);
    assert(std::rbegin(a).base() == a + 3);

    ASSERT_SAME_TYPE(decltype(std::rend(a)), std::reverse_iterator<int*>);
    assert(std::rend(a).base() == a);

    // kind of overkill since it follows from the definition, but worth testing
    ASSERT_SAME_TYPE(decltype(std::rbegin(ca)), std::reverse_iterator<const int*>);
    assert(std::rbegin(ca).base() == ca + 3);

    ASSERT_SAME_TYPE(decltype(std::rend(ca)), std::reverse_iterator<const int*>);
    assert(std::rend(ca).base() == ca);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  test_r();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif
#if TEST_STD_VER >= 17
  static_assert(test_r(), "");
#endif

  // Make sure std::begin(T (&)[N]) and std::end(T (&)[N]) are constexpr in C++11 too (see LWG2280).
  {
    static constexpr int a[] = {1, 2, 3};
    constexpr auto b         = std::begin(a);
    assert(b == a);
    constexpr auto e = std::end(a);
    assert(e == a + 3);
  }

  return 0;
}
