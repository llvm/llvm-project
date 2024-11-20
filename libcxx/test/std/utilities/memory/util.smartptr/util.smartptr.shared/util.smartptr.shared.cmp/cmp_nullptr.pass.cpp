//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template <class T>
//     bool operator==(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator==(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator!=(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator!=(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator<(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator<(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator<=(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator<=(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator>(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator>(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator>=(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator>=(nullptr_t, const shared_ptr<T>& y) noexcept;
// template<class T>
//     strong_ordering operator<=>(shared_ptr<T> const& x, nullptr_t) noexcept;   // C++20

#include <cassert>
#include <cstddef>
#include <memory>

#include "test_macros.h"
#include "test_comparisons.h"

void do_nothing(int*) {}

int main(int, char**)
{
  AssertComparisonsAreNoexcept<std::shared_ptr<int>, nullptr_t>();
  AssertComparisonsAreNoexcept<nullptr_t, std::shared_ptr<int> >();
  AssertComparisonsReturnBool<std::shared_ptr<int>, nullptr_t>();
  AssertComparisonsReturnBool<nullptr_t, std::shared_ptr<int> >();
#if TEST_STD_VER >= 20
  AssertOrderAreNoexcept<std::shared_ptr<int>>();
  AssertOrderReturn<std::strong_ordering, std::shared_ptr<int>>();
#endif

  const std::shared_ptr<int> p1(new int(1));
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
#if TEST_STD_VER >= 20
  assert((nullptr <=> p1) == std::strong_ordering::less);
  assert((p1 <=> nullptr) == std::strong_ordering::greater);
#endif

  const std::shared_ptr<int> p2;
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
#if TEST_STD_VER >= 20
  assert((p2 <=> nullptr) == std::strong_ordering::equivalent);
  assert((nullptr <=> p2) == std::strong_ordering::equivalent);
#endif

#if TEST_STD_VER >= 17
  const std::shared_ptr<int[]> p3(new int[1]);
  assert(!(p3 == nullptr));
  assert(!(nullptr == p3));
  assert(!(p3 < nullptr));
  assert((nullptr < p3));
  assert(!(p3 <= nullptr));
  assert((nullptr <= p3));
  assert((p3 > nullptr));
  assert(!(nullptr > p3));
  assert((p3 >= nullptr));
  assert(!(nullptr >= p3));
#  if TEST_STD_VER >= 20
  assert((p3 <=> nullptr) == std::strong_ordering::greater);
  assert((nullptr <=> p3) == std::strong_ordering::less);
#  endif

  const std::shared_ptr<int[]> p4;
  assert((p4 == nullptr));
  assert((nullptr == p4));
  assert(!(p4 < nullptr));
  assert(!(nullptr < p4));
  assert((p4 <= nullptr));
  assert((nullptr <= p4));
  assert(!(p4 > nullptr));
  assert(!(nullptr > p4));
  assert((p4 >= nullptr));
  assert((nullptr >= p4));
#  if TEST_STD_VER >= 20
  assert((p4 <=> nullptr) == std::strong_ordering::equivalent);
  assert((nullptr <=> p4) == std::strong_ordering::equivalent);
#  endif
#endif

  return 0;
}
