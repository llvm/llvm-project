//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr       reverse_iterator  rend() const noexcept;
// constexpr const_reverse_iterator crend() const noexcept; // since C++23

#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

template <class Span, class Iter>
constexpr void testSpanImpl(Span s, Iter rlast) {
  if (s.empty()) {
    assert(rlast == s.rbegin());
  } else {
    assert(rlast != s.rbegin());
    assert(rlast == std::make_reverse_iterator(s.begin()));
  }

  assert(static_cast<std::size_t>(rlast - s.rbegin()) == s.size());
}

template <class EType, size_t Extent, class... Args>
constexpr void testSpan(Args&&... args) {
  auto s1 = std::span<EType>(std::forward<Args>(args)...);

  testSpanImpl(s1, s1.rend());
#if TEST_STD_VER >= 23
  testSpanImpl(s1, s1.crend());
#endif

  auto s2 = std::span<EType, Extent>(std::forward<Args>(args)...);
  testSpanImpl(s2, s2.rend());
#if TEST_STD_VER >= 23
  testSpanImpl(s2, s2.crend());
#endif
}

struct A {};
bool operator==(A, A) { return true; }

constexpr int iArr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int iArr2[]           = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

constexpr bool test_runtime_and_constexpr() {
  testSpan<int, 0>();
  testSpan<long, 0>();
  testSpan<double, 0>();
  testSpan<A, 0>();
  testSpan<std::string, 0>();

  testSpan<const int, 1>(iArr1, 1);
  testSpan<const int, 2>(iArr1, 2);
  testSpan<const int, 3>(iArr1, 3);
  testSpan<const int, 4>(iArr1, 4);
  testSpan<const int, 5>(iArr1, 5);

  const std::string s2;
  testSpan<const std::string, 0>(&s2, static_cast<size_t>(0));
  testSpan<const std::string, 1>(&s2, 1);

  return true;
}

int main(int, char**) {
  test_runtime_and_constexpr();
  static_assert(test_runtime_and_constexpr());

  testSpan<int, 1>(iArr2, 1);
  testSpan<int, 2>(iArr2, 2);
  testSpan<int, 3>(iArr2, 3);
  testSpan<int, 4>(iArr2, 4);
  testSpan<int, 5>(iArr2, 5);

  std::string s1;
  testSpan<std::string, 0>(&s1, static_cast<size_t>(0));
  testSpan<std::string, 1>(&s1, 1);

  return 0;
}
