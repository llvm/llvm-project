//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// template <class T, class D>
//     constexpr bool operator==(const unique_ptr<T, D>& x, nullptr_t) noexcept; // constexpr since C++23
// template <class T, class D>
//     bool operator==(nullptr_t, const unique_ptr<T, D>& y) noexcept;           // removed in C++20
// template <class T, class D>
//     bool operator!=(const unique_ptr<T, D>& x, nullptr_t) noexcept;           // removed in C++20
// template <class T, class D>
//     bool operator!=(nullptr_t, const unique_ptr<T, D>& y) noexcept;           // removed in C++20
// template <class T, class D>
//     constexpr bool operator<(const unique_ptr<T, D>& x, nullptr_t);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator<(nullptr_t, const unique_ptr<T, D>& y);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator<=(const unique_ptr<T, D>& x, nullptr_t);          // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator<=(nullptr_t, const unique_ptr<T, D>& y);          // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>(const unique_ptr<T, D>& x, nullptr_t);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>(nullptr_t, const unique_ptr<T, D>& y);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>=(const unique_ptr<T, D>& x, nullptr_t);          // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>=(nullptr_t, const unique_ptr<T, D>& y);          // constexpr since C++23
// template<class T, class D>
//   requires three_way_comparable<typename unique_ptr<T, D>::pointer>
//   constexpr compare_three_way_result_t<typename unique_ptr<T, D>::pointer>
//     operator<=>(const unique_ptr<T, D>& x, nullptr_t);                        // C++20

#include <memory>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_comparisons.h"

TEST_CONSTEXPR_CXX23 bool test() {
  if (!TEST_IS_CONSTANT_EVALUATED) {
    AssertEqualityAreNoexcept<std::unique_ptr<int>, nullptr_t>();
    AssertEqualityAreNoexcept<nullptr_t, std::unique_ptr<int> >();
    AssertComparisonsReturnBool<std::unique_ptr<int>, nullptr_t>();
    AssertComparisonsReturnBool<nullptr_t, std::unique_ptr<int> >();
#if TEST_STD_VER > 17
    AssertOrderReturn<std::strong_ordering, std::unique_ptr<int>, nullptr_t>();
    AssertOrderReturn<std::strong_ordering, nullptr_t, std::unique_ptr<int>>();
#endif
  }

  const std::unique_ptr<int> p1(new int(1));
  assert(!(p1 == nullptr));
  assert(!(nullptr == p1));
  // A pointer to allocated storage and a nullptr can't be compared at compile-time
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(!(p1 < nullptr));
    assert((nullptr < p1));
    assert(!(p1 <= nullptr));
    assert((nullptr <= p1));
    assert((p1 > nullptr));
    assert(!(nullptr > p1));
    assert((p1 >= nullptr));
    assert(!(nullptr >= p1));
#if TEST_STD_VER > 17
    assert((nullptr <=> p1) == std::strong_ordering::less);
    assert((p1 <=> nullptr) == std::strong_ordering::greater);
#endif
  }

  const std::unique_ptr<int> p2;
  assert((p2 == nullptr));
  assert((nullptr == p2));
  assert(!(p2 < nullptr));
  assert(!(nullptr < p2));
  assert((p2 <= nullptr));
  assert((nullptr <= p2));
  assert(!(p2 > nullptr));
  assert(!(nullptr > p2));
  assert((p2 >= nullptr));
  assert((nullptr >= p2));
#if TEST_STD_VER > 17
  assert((nullptr <=> p2) == std::strong_ordering::equivalent);
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
