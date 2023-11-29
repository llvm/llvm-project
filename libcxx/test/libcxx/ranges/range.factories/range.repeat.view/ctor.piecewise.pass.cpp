//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// REQUIRES: has-unix-headers
// XFAIL: availability-verbose_abort-missing

// template<class... TArgs, class... BoundArgs>
//       requires constructible_from<T, TArgs...> &&
//                constructible_from<Bound, BoundArgs...>
//     constexpr explicit repeat_view(piecewise_construct_t,
//       tuple<TArgs...> value_args, tuple<BoundArgs...> bound_args = tuple<>{});

#include <ranges>
#include <tuple>

#include "check_assertion.h"

// clang-format off
int main(int, char**) {
  using Repeat = std::ranges::repeat_view<int, int>;
  TEST_LIBCPP_ASSERT_FAILURE(Repeat(std::piecewise_construct, std::tuple{1}, std::tuple{-1}), "The behavior is undefined if Bound is not unreachable_sentinel_t and bound is negative");

  return 0;
}
// clang-format on
