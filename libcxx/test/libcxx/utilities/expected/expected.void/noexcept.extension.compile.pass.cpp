//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test libc++ noexcept extensions on operations of std::expected<void, E>

#include <expected>
#include <type_traits>
#include <utility>

#include "../types.h"

// expected(const expected&)
static_assert(std::is_nothrow_copy_constructible_v<std::expected<void, int>>);
static_assert(!std::is_nothrow_copy_constructible_v<std::expected<void, CopyMayThrow>>);


// expected(const expected<OtherT, OtherE>&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<void, long>,
              const std::expected<void, int>&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromCopyIntMayThrow>,
              const std::expected<void, int>&>);


// expected(expected<OtherT, OtherE>&&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<void, long>,
              std::expected<void, int>&&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromMoveIntMayThrow>,
              std::expected<void, int>&&>);

// expected(const unexpected<OtherE>&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<void, long>,
              const std::unexpected<int>&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromCopyIntMayThrow>,
              const std::unexpected<int>&>);

// expected(unexpected<OtherE>&&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<void, long>,
              std::unexpected<int>&&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromMoveIntMayThrow>,
              std::unexpected<int>&&>);


// expected(unexpect_t, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::expected<void, int>,
              std::unexpect_t,
              const int&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromCopyIntMayThrow>,
              std::unexpect_t,
              const int&>);

// expected(unexpect_t, initializer_list<U>, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromInitializerListNoexcept>,
              std::unexpect_t,
              std::initializer_list<int>>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<void, ConvertFromInitializerListMayThrow>,
              std::unexpect_t,
              std::initializer_list<int>>);

// expected& operator=(const expected&)
static_assert(std::is_nothrow_copy_assignable_v<std::expected<void, int>>);
static_assert(!std::is_nothrow_copy_assignable_v<std::expected<void, CopyConstructMayThrow>>);
static_assert(!std::is_nothrow_copy_assignable_v<std::expected<void, CopyAssignMayThrow>>);
