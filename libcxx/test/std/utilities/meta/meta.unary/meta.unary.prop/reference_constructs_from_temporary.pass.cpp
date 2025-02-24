//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <type_traits>

// template<class T, class U> struct reference_constructs_from_temporary;

// template<class T, class U>
// constexpr bool reference_constructs_from_temporary_v
//   = reference_constructs_from_temporary<T, U>::value;

#include <cassert>
#include <type_traits>

struct NonPOD {
  NonPOD(int);
};
enum Enum { EV };
struct POD {
  Enum e;
  int i;
  float f;
  NonPOD* p;
};
// Not PODs
struct Derives : POD {};

template <class T, class RefType = T&>
struct ConvertsToRef {
  operator RefType() const { return static_cast<RefType>(obj); }
  mutable T obj = 42;
};
template <class T, class RefType = T&>
class ConvertsToRefPrivate {
  operator RefType() const { return static_cast<RefType>(obj); }
  mutable T obj = 42;
};

struct ExplicitConversionRvalueRef {
  operator int();
  explicit operator int&&();
};

struct ExplicitConversionRef {
  operator int();
  explicit operator int&();
};

template <typename T, typename U, bool Expected>
constexpr void test_reference_constructs_from_temporary() {
  assert((std::reference_constructs_from_temporary<T, U>::value == Expected));
  assert((std::reference_constructs_from_temporary_v<T, U> == Expected));
}

constexpr bool test() {
  test_reference_constructs_from_temporary<int&, int&, false>();
  test_reference_constructs_from_temporary<int&, int&, false>();
  test_reference_constructs_from_temporary<int&, int&&, false>();

  test_reference_constructs_from_temporary<const int&, int&, false>();
  test_reference_constructs_from_temporary<const int&, const int&, false>();
  test_reference_constructs_from_temporary<const int&, int&&, false>();

  test_reference_constructs_from_temporary<int&, long&, false>(); // doesn't construct

  test_reference_constructs_from_temporary<const int&, long&, true>();
  test_reference_constructs_from_temporary<const int&, long&&, true>();
  test_reference_constructs_from_temporary<int&&, long&, true>();

  using LRef    = ConvertsToRef<int, int&>;
  using RRef    = ConvertsToRef<int, int&&>;
  using CLRef   = ConvertsToRef<int, const int&>;
  using LongRef = ConvertsToRef<long, long&>;

  assert((std::is_constructible_v<int&, LRef>));
  test_reference_constructs_from_temporary<int&, LRef, false>();

  assert((std::is_constructible_v<int&&, RRef>));
  test_reference_constructs_from_temporary<int&&, RRef, false>();

  assert((std::is_constructible_v<const int&, CLRef>));
  test_reference_constructs_from_temporary<int&&, CLRef, false>();

  assert((std::is_constructible_v<const int&, LongRef>));
  test_reference_constructs_from_temporary<const int&, LongRef, true>();
  test_reference_constructs_from_temporary<const int&, ConvertsToRefPrivate<long, long&>, false>();

  // Test that it doesn't accept non-reference types as input.
  test_reference_constructs_from_temporary<int, long, false>();

  test_reference_constructs_from_temporary<const int&, long, true>();

  // Additional checks
  test_reference_constructs_from_temporary<POD const&, Derives, true>();
  test_reference_constructs_from_temporary<int&&, int, true>();
  test_reference_constructs_from_temporary<const int&, int, true>();
  test_reference_constructs_from_temporary<int&&, int&&, false>();
  test_reference_constructs_from_temporary<const int&, int&&, false>();
  test_reference_constructs_from_temporary<int&&, long&&, true>();
  test_reference_constructs_from_temporary<int&&, long, true>();

  test_reference_constructs_from_temporary<int&, ExplicitConversionRef, false>();
  test_reference_constructs_from_temporary<const int&, ExplicitConversionRef, false>();
  test_reference_constructs_from_temporary<int&&, ExplicitConversionRvalueRef, false>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
