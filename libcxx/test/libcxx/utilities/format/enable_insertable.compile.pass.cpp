//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <format>

#include <format>
#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <span>
#include <stack>
#include <string>
#include <string_view>

#include "test_macros.h"

template <class CharT>
struct no_value_type {
  using iterator = CharT*;
  iterator end();
  void insert(iterator, CharT*, CharT*);
};

template <class CharT>
struct no_end {
  using value_type = CharT;
  using iterator = CharT*;
  void insert(iterator, CharT*, CharT*);
};

template <class CharT>
struct no_insert {
  using value_type = CharT;
  using iterator = CharT*;
  iterator end();
};

template <class CharT>
struct no_specialization {
  using value_type = CharT;
  using iterator = CharT*;
  iterator end();
  void insert(iterator, CharT*, CharT*);
};

template <class CharT>
struct valid {
  using value_type = CharT;
  using iterator = CharT*;
  iterator end();
  void insert(iterator, CharT*, CharT*);
};

template <>
inline constexpr bool std::__format::__enable_insertable<no_value_type<char>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<no_end<char>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<no_insert<char>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<valid<char>> = true;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <>
inline constexpr bool std::__format::__enable_insertable<no_value_type<wchar_t>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<no_end<wchar_t>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<no_insert<wchar_t>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<valid<wchar_t>> = true;
#endif

static_assert(!std::__format::__insertable<no_value_type<char>>);
static_assert(!std::__format::__insertable<no_end<char>>);
static_assert(!std::__format::__insertable<no_insert<char>>);
static_assert(!std::__format::__insertable<no_specialization<char>>);
static_assert(std::__format::__insertable<valid<char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__format::__insertable<no_value_type<wchar_t>>);
static_assert(!std::__format::__insertable<no_end<wchar_t>>);
static_assert(!std::__format::__insertable<no_insert<wchar_t>>);
static_assert(!std::__format::__insertable<no_specialization<wchar_t>>);
static_assert(std::__format::__insertable<valid<wchar_t>>);
#endif

template <>
inline constexpr bool std::__format::__enable_insertable<valid<signed char>> = true;
template <>
inline constexpr bool std::__format::__enable_insertable<valid<unsigned char>> = true;

static_assert(!std::__format::__insertable<valid<signed char>>);
static_assert(!std::__format::__insertable<valid<unsigned char>>);

static_assert(std::__format::__insertable<std::string>);
static_assert(!std::__format::__insertable<std::string_view>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::__format::__insertable<std::wstring>);
static_assert(!std::__format::__insertable<std::wstring_view>);
#endif

static_assert(!std::__format::__insertable<std::array<char, 1>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__format::__insertable<std::array<wchar_t, 1>>);
#endif

static_assert(std::__format::__insertable<std::vector<char>>);
static_assert(std::__format::__insertable<std::deque<char>>);
static_assert(!std::__format::__insertable<std::forward_list<char>>);
static_assert(std::__format::__insertable<std::list<char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::__format::__insertable<std::vector<wchar_t>>);
static_assert(std::__format::__insertable<std::deque<wchar_t>>);
static_assert(!std::__format::__insertable<std::forward_list<wchar_t>>);
static_assert(std::__format::__insertable<std::list<wchar_t>>);
#endif

static_assert(!std::__format::__insertable<std::set<char>>);
static_assert(!std::__format::__insertable<std::map<char, char>>);
static_assert(!std::__format::__insertable<std::multiset<char>>);
static_assert(!std::__format::__insertable<std::multimap<char, char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__format::__insertable<std::set<wchar_t>>);
static_assert(!std::__format::__insertable<std::map<wchar_t, wchar_t>>);
static_assert(!std::__format::__insertable<std::multiset<wchar_t>>);
static_assert(!std::__format::__insertable<std::multimap<wchar_t, wchar_t>>);
#endif

static_assert(!std::__format::__insertable<std::stack<char>>);
static_assert(!std::__format::__insertable<std::queue<char>>);
static_assert(!std::__format::__insertable<std::priority_queue<char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__format::__insertable<std::stack<wchar_t>>);
static_assert(!std::__format::__insertable<std::queue<wchar_t>>);
static_assert(!std::__format::__insertable<std::priority_queue<wchar_t>>);
#endif

static_assert(!std::__format::__insertable<std::span<char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::__format::__insertable<std::span<wchar_t>>);
#endif
