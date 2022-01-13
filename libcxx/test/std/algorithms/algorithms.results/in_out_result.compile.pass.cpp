//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts, libcpp-has-no-incomplete-ranges
//
// clang-cl and cl currently don't support [[no_unique_address]]
// XFAIL: msvc

// namespace ranges {
//   template<class InputIterator, class OutputIterator>
//     struct in_out_result;
// }

#include <algorithm>

// Size optimization.
struct Empty {};
struct Empty2 {};

static_assert(sizeof(std::ranges::in_out_result<Empty, int>) == sizeof(int));
static_assert(sizeof(std::ranges::in_out_result<int, Empty>) == sizeof(int));
static_assert(sizeof(std::ranges::in_out_result<Empty, Empty2>) == sizeof(char));
