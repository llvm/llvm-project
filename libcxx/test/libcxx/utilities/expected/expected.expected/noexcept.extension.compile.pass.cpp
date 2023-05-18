//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test libc++ noexcept extensions on operations of std::expected<T, E>

#include <expected>
#include <type_traits>
#include <utility>

#include "../types.h"

// expected();
static_assert(std::is_nothrow_default_constructible_v<std::expected<int, int>>);
static_assert(!std::is_nothrow_default_constructible_v<std::expected<DefaultMayThrow, int>>);

// expected(const expected&)
static_assert(std::is_nothrow_copy_constructible_v<std::expected<int, int>>);
static_assert(!std::is_nothrow_copy_constructible_v<std::expected<CopyMayThrow, int>>);
static_assert(!std::is_nothrow_copy_constructible_v<std::expected<int, CopyMayThrow>>);

// expected(const expected<OtherT, OtherE>&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<long, long>,
              const std::expected<int, int>&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<ConvertFromCopyIntMayThrow, long>,
              const std::expected<int, int>&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<long, ConvertFromCopyIntMayThrow>,
              const std::expected<int, int>&>);
static_assert(!std::is_nothrow_constructible_v<                                      //
              std::expected<ConvertFromCopyIntMayThrow, ConvertFromCopyIntMayThrow>, //
              const std::expected<int, int>&>);

// expected(expected<OtherT, OtherE>&&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<long, long>,
              std::expected<int, int>&&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<ConvertFromMoveIntMayThrow, long>,
              std::expected<int, int>&&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<long, ConvertFromMoveIntMayThrow>,
              std::expected<int, int>&&>);
static_assert(!std::is_nothrow_constructible_v<                                      //
              std::expected<ConvertFromMoveIntMayThrow, ConvertFromMoveIntMayThrow>, //
              std::expected<int, int>&&>);

// expected(U&&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<int, int>,
              const int&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<ConvertFromCopyIntMayThrow, int>,
              const int&>);

// expected(const unexpected<OtherE>&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<int, long>,
              const std::unexpected<int>&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<int, ConvertFromCopyIntMayThrow>,
              const std::unexpected<int>&>);

// expected(unexpected<OtherE>&&)
static_assert(std::is_nothrow_constructible_v< //
              std::expected<int, long>,
              std::unexpected<int>&&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<int, ConvertFromMoveIntMayThrow>,
              std::unexpected<int>&&>);

// expected(in_place_t, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::expected<int, int>,
              std::in_place_t,
              const int&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<ConvertFromCopyIntMayThrow, int>,
              std::in_place_t,
              const int&>);

// expected(in_place_t, initializer_list<U>, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::expected<ConvertFromInitializerListNoexcept, int>,
              std::in_place_t,
              std::initializer_list<int>>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<ConvertFromInitializerListMayThrow, int>,
              std::in_place_t,
              std::initializer_list<int>>);

// expected(unexpect_t, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::expected<int, int>,
              std::unexpect_t,
              const int&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<int, ConvertFromCopyIntMayThrow>,
              std::unexpect_t,
              const int&>);

// expected(unexpect_t, initializer_list<U>, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::expected<int, ConvertFromInitializerListNoexcept>,
              std::unexpect_t,
              std::initializer_list<int>>);
static_assert(!std::is_nothrow_constructible_v< //
              std::expected<int, ConvertFromInitializerListMayThrow>,
              std::unexpect_t,
              std::initializer_list<int>>);

// expected& operator=(const expected&)
static_assert(std::is_nothrow_copy_assignable_v<std::expected<int, int>>);
static_assert(!std::is_nothrow_copy_assignable_v<std::expected<CopyConstructMayThrow, int>>);
static_assert(!std::is_nothrow_copy_assignable_v<std::expected<CopyAssignMayThrow, int>>);
static_assert(!std::is_nothrow_copy_assignable_v<std::expected<int, CopyConstructMayThrow>>);
static_assert(!std::is_nothrow_copy_assignable_v<std::expected<int, CopyAssignMayThrow>>);
