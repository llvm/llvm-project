//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// Instantiating std::tuple<T...> must not form tuple_size_v of the tuple itself, as the tuple-like
// operator<=> (C++23) and operator== (C++26) once did, which instantiated user tuple_size
// specializations against the still-incomplete tuple.

// REQUIRES: std-at-least-c++23

#include <tuple>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

struct Base {};

template <std::derived_from<Base> T>
struct std::tuple_size<T> : std::integral_constant<std::size_t, 1> {};

void test() {
  std::tuple<int> t = {1};
  std::array<int, 1> a{0};
  (void)(t <=> a);
  (void)(t == a);
}
