//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// ~tuple();

// C++17 added:
//   The destructor of tuple shall be a trivial destructor
//     if (is_trivially_destructible_v<Types> && ...) is true.

#include <tuple>
#include <string>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(std::is_trivially_destructible<
      std::tuple<> >::value, "");
  static_assert(std::is_trivially_destructible<
      std::tuple<void*> >::value, "");
  static_assert(std::is_trivially_destructible<
      std::tuple<int, float> >::value, "");
  static_assert(!std::is_trivially_destructible<
      std::tuple<std::string> >::value, "");
  static_assert(!std::is_trivially_destructible<
      std::tuple<int, std::string> >::value, "");

  return 0;
}
