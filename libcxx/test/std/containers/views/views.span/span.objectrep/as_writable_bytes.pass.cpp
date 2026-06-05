//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// template<class ElementType, size_t Extent>
//   span<byte, Extent == dynamic_extent ? dynamic_extent : sizeof(ElementType) * Extent>
//     as_writable_bytes(span<ElementType, Extent> s) noexcept;
//
// Constraints:
//   is_const_v<ElementType> is false and is_volatile_v<ElementType> is false.

#include <cassert>
#include <cstddef>
#include <span>
#include <string>

#include "test_macros.h"

template <class T, std::size_t Extent = std::dynamic_extent>
concept hasAsWritableBytes = requires(std::span<T, Extent> s) { std::as_writable_bytes(s); };

template <typename Span>
void testRuntimeSpan(Span sp) {
  ASSERT_NOEXCEPT(std::as_writable_bytes(sp));

  auto spBytes = std::as_writable_bytes(sp);
  using SB     = decltype(spBytes);
  ASSERT_SAME_TYPE(std::byte, typename SB::element_type);

  if constexpr (sp.extent == std::dynamic_extent)
    assert(spBytes.extent == std::dynamic_extent);
  else
    assert(spBytes.extent == sizeof(typename Span::element_type) * sp.extent);

  assert(static_cast<void*>(spBytes.data()) == static_cast<void*>(sp.data()));
  assert(spBytes.size() == sp.size_bytes());
}

struct A {};
int iArr2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

void test_constraints() {
  static_assert(hasAsWritableBytes<int>);
  static_assert(hasAsWritableBytes<long>);
  static_assert(hasAsWritableBytes<double>);
  static_assert(hasAsWritableBytes<A>);
  static_assert(hasAsWritableBytes<std::string>);

  static_assert(!hasAsWritableBytes<const int>);
  static_assert(!hasAsWritableBytes<const long>);
  static_assert(!hasAsWritableBytes<const double>);
  static_assert(!hasAsWritableBytes<const A>);
  static_assert(!hasAsWritableBytes<const std::string>);

  static_assert(!hasAsWritableBytes<volatile int>);
  static_assert(!hasAsWritableBytes<volatile long>);
  static_assert(!hasAsWritableBytes<volatile double>);
  static_assert(!hasAsWritableBytes<volatile A>);
  static_assert(!hasAsWritableBytes<volatile std::string>);

  static_assert(!hasAsWritableBytes<const volatile int>);
  static_assert(!hasAsWritableBytes<const volatile long>);
  static_assert(!hasAsWritableBytes<const volatile double>);
  static_assert(!hasAsWritableBytes<const volatile A>);
  static_assert(!hasAsWritableBytes<const volatile std::string>);

  static_assert(hasAsWritableBytes<int, 0>);
  static_assert(hasAsWritableBytes<long, 0>);
  static_assert(hasAsWritableBytes<double, 0>);
  static_assert(hasAsWritableBytes<A, 0>);
  static_assert(hasAsWritableBytes<std::string, 0>);

  static_assert(!hasAsWritableBytes<const int, 0>);
  static_assert(!hasAsWritableBytes<const long, 0>);
  static_assert(!hasAsWritableBytes<const double, 0>);
  static_assert(!hasAsWritableBytes<const A, 0>);
  static_assert(!hasAsWritableBytes<const std::string, 0>);

  static_assert(!hasAsWritableBytes<volatile int, 0>);
  static_assert(!hasAsWritableBytes<volatile long, 0>);
  static_assert(!hasAsWritableBytes<volatile double, 0>);
  static_assert(!hasAsWritableBytes<volatile A, 0>);
  static_assert(!hasAsWritableBytes<volatile std::string, 0>);

  static_assert(!hasAsWritableBytes<const volatile int, 0>);
  static_assert(!hasAsWritableBytes<const volatile long, 0>);
  static_assert(!hasAsWritableBytes<const volatile double, 0>);
  static_assert(!hasAsWritableBytes<const volatile A, 0>);
  static_assert(!hasAsWritableBytes<const volatile std::string, 0>);
}

int main() {
  test_constraints();

  testRuntimeSpan(std::span<int>());
  testRuntimeSpan(std::span<long>());
  testRuntimeSpan(std::span<double>());
  testRuntimeSpan(std::span<A>());
  testRuntimeSpan(std::span<std::string>());

  testRuntimeSpan(std::span<int, 0>());
  testRuntimeSpan(std::span<long, 0>());
  testRuntimeSpan(std::span<double, 0>());
  testRuntimeSpan(std::span<A, 0>());
  testRuntimeSpan(std::span<std::string, 0>());

  testRuntimeSpan(std::span<int>(iArr2, 1));
  testRuntimeSpan(std::span<int>(iArr2, 2));
  testRuntimeSpan(std::span<int>(iArr2, 3));
  testRuntimeSpan(std::span<int>(iArr2, 4));
  testRuntimeSpan(std::span<int>(iArr2, 5));

  testRuntimeSpan(std::span<int, 1>(iArr2 + 5, 1));
  testRuntimeSpan(std::span<int, 2>(iArr2 + 4, 2));
  testRuntimeSpan(std::span<int, 3>(iArr2 + 3, 3));
  testRuntimeSpan(std::span<int, 4>(iArr2 + 2, 4));
  testRuntimeSpan(std::span<int, 5>(iArr2 + 1, 5));

  std::string s;
  testRuntimeSpan(std::span<std::string>(&s, (std::size_t)0));
  testRuntimeSpan(std::span<std::string>(&s, 1));

  return 0;
}
