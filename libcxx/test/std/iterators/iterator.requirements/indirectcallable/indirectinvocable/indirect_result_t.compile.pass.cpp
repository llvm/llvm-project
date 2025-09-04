//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// indirect_result_t

#include <iterator>
#include <concepts>

#include "test_macros.h"

static_assert(std::same_as<std::indirect_result_t<int (*)(int), int*>, int>);
static_assert(std::same_as<std::indirect_result_t<double (*)(int const&, float), int const*, float*>, double>);

struct S { };
static_assert(std::same_as<std::indirect_result_t<S (&)(int), int*>, S>);
static_assert(std::same_as<std::indirect_result_t<long S::*, S*>, long&>);
static_assert(std::same_as<std::indirect_result_t<S && (S::*)(), S*>, S&&>);
static_assert(std::same_as<std::indirect_result_t<int S::* (S::*)(int) const, S*, int*>, int S::*>);

template <class F, class... Is>
constexpr bool has_indirect_result = requires {
  typename std::indirect_result_t<F, Is...>;
};

static_assert(!has_indirect_result<int (*)(int), int>); // int isn't indirectly_readable
static_assert(!has_indirect_result<int, int*>);         // int isn't invocable

// Test ADL-proofing (P2538R1)
#if TEST_STD_VER >= 26 || defined(_LIBCPP_VERSION)
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(std::same_as<std::indirect_result_t<int                (&)(int), int*>, int>);
static_assert(std::same_as<std::indirect_result_t<Holder<Incomplete>&(&)(int), int*>, Holder<Incomplete>&>);
static_assert(std::same_as<std::indirect_result_t<Holder<Incomplete>*(&)(int), int*>, Holder<Incomplete>*>);
static_assert(std::same_as<std::indirect_result_t<int                (&)(Holder<Incomplete>*), Holder<Incomplete>**>, int>);
static_assert(std::same_as<std::indirect_result_t<Holder<Incomplete>&(&)(Holder<Incomplete>*), Holder<Incomplete>**>, Holder<Incomplete>&>);
static_assert(std::same_as<std::indirect_result_t<Holder<Incomplete>*(&)(Holder<Incomplete>*), Holder<Incomplete>**>, Holder<Incomplete>*>);
#endif
