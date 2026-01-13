//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <ranges>

// template<class T>
// inline constexpr bool enable_view = ...;

#include <ranges>

#include "test_macros.h"

// Doesn't derive from view_base
struct Empty { };
static_assert(!std::ranges::enable_view<Empty>);
static_assert(!std::ranges::enable_view<Empty&>);
static_assert(!std::ranges::enable_view<Empty&&>);
static_assert(!std::ranges::enable_view<const Empty>);
static_assert(!std::ranges::enable_view<const Empty&>);
static_assert(!std::ranges::enable_view<const Empty&&>);
static_assert(!std::ranges::enable_view<volatile Empty>);
static_assert(!std::ranges::enable_view<volatile Empty&>);
static_assert(!std::ranges::enable_view<volatile Empty&&>);
static_assert(!std::ranges::enable_view<const volatile Empty>);
static_assert(!std::ranges::enable_view<const volatile Empty&>);
static_assert(!std::ranges::enable_view<const volatile Empty&&>);

// Derives from view_base, but privately
struct PrivateViewBase : private std::ranges::view_base { };
static_assert(!std::ranges::enable_view<PrivateViewBase>);
static_assert(!std::ranges::enable_view<PrivateViewBase&>);
static_assert(!std::ranges::enable_view<PrivateViewBase&&>);
static_assert(!std::ranges::enable_view<const PrivateViewBase>);
static_assert(!std::ranges::enable_view<const PrivateViewBase&>);
static_assert(!std::ranges::enable_view<const PrivateViewBase&&>);
static_assert(!std::ranges::enable_view<volatile PrivateViewBase>);
static_assert(!std::ranges::enable_view<volatile PrivateViewBase&>);
static_assert(!std::ranges::enable_view<volatile PrivateViewBase&&>);
static_assert(!std::ranges::enable_view<const volatile PrivateViewBase>);
static_assert(!std::ranges::enable_view<const volatile PrivateViewBase&>);
static_assert(!std::ranges::enable_view<const volatile PrivateViewBase&&>);

// Derives from view_base, but specializes enable_view to false
struct EnableViewFalse : std::ranges::view_base { };
template <> constexpr bool std::ranges::enable_view<EnableViewFalse> = false;
static_assert(!std::ranges::enable_view<EnableViewFalse>);
static_assert(!std::ranges::enable_view<EnableViewFalse&>);
static_assert(!std::ranges::enable_view<EnableViewFalse&&>);
static_assert(std::ranges::enable_view<const EnableViewFalse>);
static_assert(!std::ranges::enable_view<const EnableViewFalse&>);
static_assert(!std::ranges::enable_view<const EnableViewFalse&&>);
static_assert(std::ranges::enable_view<volatile EnableViewFalse>);
static_assert(!std::ranges::enable_view<volatile EnableViewFalse&>);
static_assert(!std::ranges::enable_view<volatile EnableViewFalse&&>);
static_assert(std::ranges::enable_view<const volatile EnableViewFalse>);
static_assert(!std::ranges::enable_view<const volatile EnableViewFalse&>);
static_assert(!std::ranges::enable_view<const volatile EnableViewFalse&&>);

// Derives from view_base
struct PublicViewBase : std::ranges::view_base { };
static_assert(std::ranges::enable_view<PublicViewBase>);
static_assert(!std::ranges::enable_view<PublicViewBase&>);
static_assert(!std::ranges::enable_view<PublicViewBase&&>);
static_assert(std::ranges::enable_view<const PublicViewBase>);
static_assert(!std::ranges::enable_view<const PublicViewBase&>);
static_assert(!std::ranges::enable_view<const PublicViewBase&&>);
static_assert(std::ranges::enable_view<volatile PublicViewBase>);
static_assert(!std::ranges::enable_view<volatile PublicViewBase&>);
static_assert(!std::ranges::enable_view<volatile PublicViewBase&&>);
static_assert(std::ranges::enable_view<const volatile PublicViewBase>);
static_assert(!std::ranges::enable_view<const volatile PublicViewBase&>);
static_assert(!std::ranges::enable_view<const volatile PublicViewBase&&>);

// Does not derive from view_base, but specializes enable_view to true
struct EnableViewTrue { };
template <> constexpr bool std::ranges::enable_view<EnableViewTrue> = true;
static_assert(std::ranges::enable_view<EnableViewTrue>);
static_assert(!std::ranges::enable_view<EnableViewTrue&>);
static_assert(!std::ranges::enable_view<EnableViewTrue&&>);
static_assert(!std::ranges::enable_view<const EnableViewTrue>);
static_assert(!std::ranges::enable_view<const EnableViewTrue&>);
static_assert(!std::ranges::enable_view<const EnableViewTrue&&>);
static_assert(!std::ranges::enable_view<volatile EnableViewTrue>);
static_assert(!std::ranges::enable_view<volatile EnableViewTrue&>);
static_assert(!std::ranges::enable_view<volatile EnableViewTrue&&>);
static_assert(!std::ranges::enable_view<const volatile EnableViewTrue>);
static_assert(!std::ranges::enable_view<const volatile EnableViewTrue&>);
static_assert(!std::ranges::enable_view<const volatile EnableViewTrue&&>);

// Make sure that enable_view is a bool, not some other contextually-convertible-to-bool type.
ASSERT_SAME_TYPE(decltype(std::ranges::enable_view<Empty>), const bool);
ASSERT_SAME_TYPE(decltype(std::ranges::enable_view<PublicViewBase>), const bool);

struct V1 : std::ranges::view_interface<V1> {};
static_assert(std::ranges::enable_view<V1>);
static_assert(!std::ranges::enable_view<V1&>);
static_assert(!std::ranges::enable_view<V1&&>);
static_assert(std::ranges::enable_view<const V1>);
static_assert(!std::ranges::enable_view<const V1&>);
static_assert(!std::ranges::enable_view<const V1&&>);
static_assert(std::ranges::enable_view<volatile V1>);
static_assert(!std::ranges::enable_view<volatile V1&>);
static_assert(!std::ranges::enable_view<volatile V1&&>);
static_assert(std::ranges::enable_view<const volatile V1>);
static_assert(!std::ranges::enable_view<const volatile V1&>);
static_assert(!std::ranges::enable_view<const volatile V1&&>);

struct V2 : std::ranges::view_interface<V1>, std::ranges::view_interface<V2> {};
static_assert(!std::ranges::enable_view<V2>);
static_assert(!std::ranges::enable_view<V2&>);
static_assert(!std::ranges::enable_view<V2&&>);
static_assert(!std::ranges::enable_view<const V2>);
static_assert(!std::ranges::enable_view<const V2&>);
static_assert(!std::ranges::enable_view<const V2&&>);
static_assert(!std::ranges::enable_view<volatile V2>);
static_assert(!std::ranges::enable_view<volatile V2&>);
static_assert(!std::ranges::enable_view<volatile V2&&>);
static_assert(!std::ranges::enable_view<const volatile V2>);
static_assert(!std::ranges::enable_view<const volatile V2&>);
static_assert(!std::ranges::enable_view<const volatile V2&&>);

struct V3 : std::ranges::view_interface<V1> {};
static_assert(std::ranges::enable_view<V3>);
static_assert(!std::ranges::enable_view<V3&>);
static_assert(!std::ranges::enable_view<V3&&>);
static_assert(std::ranges::enable_view<const V3>);
static_assert(!std::ranges::enable_view<const V3&>);
static_assert(!std::ranges::enable_view<const V3&&>);
static_assert(std::ranges::enable_view<volatile V3>);
static_assert(!std::ranges::enable_view<volatile V3&>);
static_assert(!std::ranges::enable_view<volatile V3&&>);
static_assert(std::ranges::enable_view<const volatile V3>);
static_assert(!std::ranges::enable_view<const volatile V3&>);
static_assert(!std::ranges::enable_view<const volatile V3&&>);

struct PrivateInherit : private std::ranges::view_interface<PrivateInherit> {};
static_assert(!std::ranges::enable_view<PrivateInherit>);
static_assert(!std::ranges::enable_view<const PrivateInherit>);
static_assert(!std::ranges::enable_view<volatile PrivateInherit>);
static_assert(!std::ranges::enable_view<const volatile PrivateInherit>);

// https://llvm.org/PR132577
// enable_view<view_interface<T>> should be false.
static_assert(!std::ranges::enable_view<std::ranges::view_interface<V1>>);
static_assert(!std::ranges::enable_view<const std::ranges::view_interface<V1>>);
static_assert(!std::ranges::enable_view<volatile std::ranges::view_interface<V1>>);
static_assert(!std::ranges::enable_view<const volatile std::ranges::view_interface<V1>>);

// ADL-proof
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::ranges::enable_view<Holder<Incomplete>*>);
static_assert(!std::ranges::enable_view<const Holder<Incomplete>*>);
static_assert(!std::ranges::enable_view<volatile Holder<Incomplete>*>);
static_assert(!std::ranges::enable_view<const volatile Holder<Incomplete>*>);

static_assert(!std::ranges::enable_view<void>);
static_assert(!std::ranges::enable_view<const void>);
static_assert(!std::ranges::enable_view<volatile void>);
static_assert(!std::ranges::enable_view<const volatile void>);
