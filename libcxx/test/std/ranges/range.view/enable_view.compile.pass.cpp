//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <ranges>

// template<class T>
// inline constexpr bool enable_view = ...;

#include <ranges>

#include "test_macros.h"

// Doesn't derive from view_base
struct Empty { };
static_assert(!std::ranges::enable_view<Empty>);

// Derives from view_base, but privately
struct PrivateViewBase : private std::ranges::view_base { };
static_assert(!std::ranges::enable_view<PrivateViewBase>);

// Derives from view_base, but specializes enable_view to false
struct EnableViewFalse : std::ranges::view_base { };
namespace std::ranges { template <> constexpr bool enable_view<EnableViewFalse> = false; }
static_assert(!std::ranges::enable_view<EnableViewFalse>);


// Derives from view_base
struct PublicViewBase : std::ranges::view_base { };
static_assert(std::ranges::enable_view<PublicViewBase>);

// Does not derive from view_base, but specializes enable_view to true
struct EnableViewTrue { };
namespace std::ranges { template <> constexpr bool enable_view<EnableViewTrue> = true; }
static_assert(std::ranges::enable_view<EnableViewTrue>);


// Make sure that enable_view is a bool, not some other contextually-convertible-to-bool type.
ASSERT_SAME_TYPE(decltype(std::ranges::enable_view<Empty>), const bool);
ASSERT_SAME_TYPE(decltype(std::ranges::enable_view<PublicViewBase>), const bool);
