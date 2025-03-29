//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>, <iterator>, <ranges>

// ADL should be performed. Ordinary unqualified lookup should not be performed.

namespace ns {
struct StructWithGlobalFunctions {};
} // namespace ns

struct ConvertibleToCmpType;
ConvertibleToCmpType strong_order(const ns::StructWithGlobalFunctions&, const ns::StructWithGlobalFunctions&);
ConvertibleToCmpType weak_order(const ns::StructWithGlobalFunctions&, const ns::StructWithGlobalFunctions&);
ConvertibleToCmpType partial_order(const ns::StructWithGlobalFunctions&, const ns::StructWithGlobalFunctions&);

int&& iter_move(const ns::StructWithGlobalFunctions&);
void iter_swap(const ns::StructWithGlobalFunctions&, const ns::StructWithGlobalFunctions&);

int* begin(const ns::StructWithGlobalFunctions&);
int* end(const ns::StructWithGlobalFunctions&);
int* rbegin(const ns::StructWithGlobalFunctions&);
int* rend(const ns::StructWithGlobalFunctions&);
unsigned int size(const ns::StructWithGlobalFunctions&);

#include <compare>
#include <ranges>
#include <type_traits>

struct ConvertibleToCmpType {
  operator std::strong_ordering() const;
  operator std::weak_ordering() const;
  operator std::partial_ordering() const;
};

struct StructWithHiddenFriends {
  friend ConvertibleToCmpType strong_order(const StructWithHiddenFriends&, const StructWithHiddenFriends&);
  friend ConvertibleToCmpType weak_order(const StructWithHiddenFriends&, const StructWithHiddenFriends&);
  friend ConvertibleToCmpType partial_order(const StructWithHiddenFriends&, const StructWithHiddenFriends&);

  friend int&& iter_move(const StructWithHiddenFriends&);
  friend void iter_swap(const StructWithHiddenFriends&, const StructWithHiddenFriends&);

  friend int* begin(const StructWithHiddenFriends&);
  friend int* end(const StructWithHiddenFriends&);
  friend int* rbegin(const StructWithHiddenFriends&);
  friend int* rend(const StructWithHiddenFriends&);
  friend unsigned int size(const StructWithHiddenFriends&);
};

// [cmp.alg] ADL should be performed.
static_assert(std::is_invocable_v<decltype(std::strong_order), StructWithHiddenFriends&, StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::weak_order), StructWithHiddenFriends&, StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::partial_order), StructWithHiddenFriends&, StructWithHiddenFriends&>);

// [cmp.alg] Ordinary unqualified lookup should not be performed.
static_assert(
    !std::is_invocable_v<decltype(std::strong_order), ns::StructWithGlobalFunctions&, ns::StructWithGlobalFunctions&>);
static_assert(
    !std::is_invocable_v<decltype(std::weak_order), ns::StructWithGlobalFunctions&, ns::StructWithGlobalFunctions&>);
static_assert(
    !std::is_invocable_v<decltype(std::partial_order), ns::StructWithGlobalFunctions&, ns::StructWithGlobalFunctions&>);

// [iterator.cust] ADL should be performed.
static_assert(std::is_invocable_v<decltype(std::ranges::iter_move), StructWithHiddenFriends&>);
static_assert(
    std::is_invocable_v<decltype(std::ranges::iter_swap), StructWithHiddenFriends&, StructWithHiddenFriends&>);

// [iterator.cust] Ordinary unqualified lookup should not be performed.
static_assert(!std::is_invocable_v<decltype(std::ranges::iter_move), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::iter_swap),
                                   ns::StructWithGlobalFunctions&,
                                   ns::StructWithGlobalFunctions&>);

// [range.access] ADL should be performed.
static_assert(std::is_invocable_v<decltype(std::ranges::begin), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::cbegin), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::end), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::cend), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::rbegin), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::crbegin), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::rend), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::crend), StructWithHiddenFriends&>);
static_assert(std::is_invocable_v<decltype(std::ranges::size), StructWithHiddenFriends&>);

// [range.access] Ordinary unqualified lookup should not be performed.
static_assert(!std::is_invocable_v<decltype(std::ranges::begin), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::cbegin), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::end), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::cend), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::rbegin), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::crbegin), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::rend), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::crend), ns::StructWithGlobalFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::size), ns::StructWithGlobalFunctions&>);
