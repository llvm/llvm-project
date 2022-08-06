//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template <class T, class D>
//     bool operator==(const unique_ptr<T, D>& x, nullptr_t) noexcept;
// template <class T, class D>
//     bool operator==(nullptr_t, const unique_ptr<T, D>& y) noexcept;
// template <class T, class D>
//     bool operator!=(const unique_ptr<T, D>& x, nullptr_t) noexcept;
// template <class T, class D>
//     bool operator!=(nullptr_t, const unique_ptr<T, D>& y) noexcept;
// template <class T, class D>
//     bool operator<(const unique_ptr<T, D>& x, nullptr_t);
// template <class T, class D>
//     bool operator<(nullptr_t, const unique_ptr<T, D>& y);
// template <class T, class D>
//     bool operator<=(const unique_ptr<T, D>& x, nullptr_t);
// template <class T, class D>
//     bool operator<=(nullptr_t, const unique_ptr<T, D>& y);
// template <class T, class D>
//     bool operator>(const unique_ptr<T, D>& x, nullptr_t);
// template <class T, class D>
//     bool operator>(nullptr_t, const unique_ptr<T, D>& y);
// template <class T, class D>
//     bool operator>=(const unique_ptr<T, D>& x, nullptr_t);
// template <class T, class D>
//     bool operator>=(nullptr_t, const unique_ptr<T, D>& y);
// template<class T, class D>
//   requires three_­way_­comparable<typename unique_ptr<T, D>::pointer>
//   constexpr compare_three_way_result_t<typename unique_ptr<T, D>::pointer>
//     operator<=>(const unique_ptr<T, D>& x, nullptr_t);                            // C++20

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
  AssertEqualityAreNoexcept<std::unique_ptr<int>, nullptr_t>();
  AssertEqualityAreNoexcept<nullptr_t, std::unique_ptr<int> >();
  AssertComparisonsReturnBool<std::unique_ptr<int>, nullptr_t>();
  AssertComparisonsReturnBool<nullptr_t, std::unique_ptr<int> >();
#if TEST_STD_VER > 17
  AssertOrderReturn<std::strong_ordering, std::unique_ptr<int>, nullptr_t>();
  AssertOrderReturn<std::strong_ordering, nullptr_t, std::unique_ptr<int>>();
#endif

  const std::unique_ptr<int> p1(new int(1));
  assert(!(p1 == nullptr));
  assert(!(nullptr == p1));
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

  return 0;
}
