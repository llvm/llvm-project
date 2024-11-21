//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test libc++ noexcept extensions on operations of std::unexpected<E>

#include <expected>
#include <type_traits>
#include <utility>

#include "../types.h"

// unexpected(Error&&);
static_assert(std::is_nothrow_constructible_v< //
              std::unexpected<int>,
              const int&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::unexpected<ConvertFromCopyIntMayThrow>,
              const int&>);


// unexpected(in_place_t, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::unexpected<int>,
              std::in_place_t,
              const int&>);
static_assert(!std::is_nothrow_constructible_v< //
              std::unexpected<ConvertFromCopyIntMayThrow>,
              std::in_place_t,
              const int&>);

// unexpected(in_place_t, initializer_list<U>, _Args&&...);
static_assert(std::is_nothrow_constructible_v< //
              std::unexpected<ConvertFromInitializerListNoexcept>,
              std::in_place_t,
              std::initializer_list<int>>);
static_assert(!std::is_nothrow_constructible_v< //
              std::unexpected<ConvertFromInitializerListMayThrow>,
              std::in_place_t,
              std::initializer_list<int>>);
