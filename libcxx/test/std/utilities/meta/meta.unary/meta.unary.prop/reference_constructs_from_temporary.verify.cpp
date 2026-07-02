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

// expected-error@*:* 2 {{incomplete type 'IncompleteType' used in type trait expression}}

#include <type_traits>

struct NoConv {};
struct Bad {
  template <class T>
  Bad(T v) noexcept(noexcept(member_ = v)) {}
  int member_;
};
struct IncompleteType;

constexpr bool test() {
  static_assert(!std::reference_constructs_from_temporary_v<IncompleteType, NoConv&&>);
  static_assert(!std::reference_constructs_from_temporary_v<Bad, IncompleteType>);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
