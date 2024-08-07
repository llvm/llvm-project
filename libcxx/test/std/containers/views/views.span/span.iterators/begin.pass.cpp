//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr       iterator  begin() const noexcept;
// constexpr const_iterator cbegin() const noexcept; // since C++23

#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

template <class Span, class Iter>
constexpr bool testSpanImpl(Span s, Iter first) {
  bool ret = true;
  if (s.empty()) {
    ret = ret && (first == s.end());
  } else {
    ret = ret && (*first == s[0]);
    ret = ret && (&*first == &s[0]);
  }
  return ret;
}

template <class EType, size_t Extent, class... Args>
constexpr bool testSpan(Args&&... args) {
  auto s1  = std::span<EType>(std::forward<Args>(args)...);
  bool ret = true;

  ret = ret && testSpanImpl(s1, s1.begin());
#if TEST_STD_VER >= 23
  ret = ret && testSpanImpl(s1, s1.cbegin());
#endif

  auto s2 = std::span<EType, Extent>(std::forward<Args>(args)...);
  ret     = ret && testSpanImpl(s2, s2.begin());
#if TEST_STD_VER >= 23
  ret = ret && testSpanImpl(s2, s2.cbegin());
#endif

  return ret;
}

struct A {};
bool operator==(A, A) { return true; }

constexpr int iArr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int iArr2[]           = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**) {
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<int, 0>());
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<long, 0>());
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<double, 0>());
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<A, 0>());
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<std::string, 0>());

  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const int, 1>(iArr1, 1));
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const int, 2>(iArr1, 2));
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const int, 3>(iArr1, 3));
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const int, 4>(iArr1, 4));
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const int, 5>(iArr1, 5));

  testSpan<int, 1>(iArr2, 1);
  testSpan<int, 2>(iArr2, 2);
  testSpan<int, 3>(iArr2, 3);
  testSpan<int, 4>(iArr2, 4);
  testSpan<int, 5>(iArr2, 5);

  std::string s1;
  constexpr static std::string s2;
  testSpan<std::string, 0>(&s1, static_cast<size_t>(0));
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const std::string, 0>(&s2, static_cast<size_t>(0)));
  testSpan<std::string, 1>(&s1, 1);
  ASSERT_RUNTIME_AND_CONSTEXPR(testSpan<const std::string, 1>(&s2, 1));

  return 0;
}
