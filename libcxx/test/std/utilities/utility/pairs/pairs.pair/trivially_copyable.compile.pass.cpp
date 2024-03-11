//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <utility>

// template <class T1, class T2> struct pair

// Checks that `std::pair` is actually trivially copyable.

#include <type_traits>
#include <utility>

#if defined(_LIBCPP_ABI_PAIR_TRIVIALLY_COPYABLE)
static_assert(std::is_trivially_copyable_v<std::pair<int, int>>);
static_assert(std::is_trivially_copyable_v<std::pair<int, int const>>);
static_assert(std::is_trivially_copyable_v<std::pair<int const, int>>);
static_assert(std::is_trivially_copyable_v<std::pair<int const, int const>>);

static_assert(!std::is_trivially_copyable_v<std::pair<int, int&>>);
static_assert(!std::is_trivially_copyable_v<std::pair<int&, int>>);
static_assert(!std::is_trivially_copyable_v<std::pair<int&, int&>>);

enum E {};
static_assert(std::is_trivially_copyable_v<std::pair<E, E>>);
static_assert(std::is_trivially_copyable_v<std::pair<E, E const>>);
static_assert(std::is_trivially_copyable_v<std::pair<E const, E>>);
static_assert(std::is_trivially_copyable_v<std::pair<E const, E const>>);

static_assert(std::is_trivially_copyable_v<std::pair<E, int>>);
static_assert(std::is_trivially_copyable_v<std::pair<E, int const>>);
static_assert(std::is_trivially_copyable_v<std::pair<E const, int>>);
static_assert(std::is_trivially_copyable_v<std::pair<E const, int const>>);

struct S {};
static_assert(std::is_trivially_copyable_v<std::pair<S, S>>);
static_assert(std::is_trivially_copyable_v<std::pair<S, int>>);
static_assert(std::is_trivially_copyable_v<std::pair<S, E>>);
#endif
