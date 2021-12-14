//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class S, class I>
// concept __nothrow_sentinel_for;

#include <memory>

static_assert(std::ranges::__nothrow_sentinel_for<int*, int*>);
static_assert(!std::ranges::__nothrow_sentinel_for<int*, long*>);

// Because `__nothrow_sentinel_for` is essentially an alias for `sentinel_for`,
// the two concepts should subsume each other.

constexpr bool ntsf_subsumes_sf(std::ranges::__nothrow_sentinel_for<char*> auto) requires true {
  return true;
}
constexpr bool ntsf_subsumes_sf(std::sentinel_for<char*> auto);

static_assert(ntsf_subsumes_sf("foo"));

constexpr bool sf_subsumes_ntsf(std::sentinel_for<char*> auto) requires true {
  return true;
}
constexpr bool sf_subsumes_ntsf(std::ranges::__nothrow_sentinel_for<char*> auto);

static_assert(sf_subsumes_ntsf("foo"));
