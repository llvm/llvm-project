//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// This test uses std::filesystem::path, which is not always available
// XFAIL: availability-filesystem-missing

// <format>

// template<ranges::input_range R>
//     requires same_as<R, remove_cvref_t<R>>
//  constexpr range_format format_kind<R> = see below;

#include <format>

#include <array>
#include <deque>
#include <filesystem>
#include <format>
#include <forward_list>
#include <iterator>
#include <list>
#include <map>
#include <ranges>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <valarray>
#include <vector>

#include "test_macros.h"

// [format.range.fmtkind]
// If same_as<remove_cvref_t<ranges::range_reference_t<R>>, R> is true,
// format_kind<R> is range_format::disabled.
// [Note 1: This prevents constraint recursion for ranges whose reference type
// is the same range type. For example, std::filesystem::path is a range of
// std::filesystem::path. - end note]
struct recursive_range {
  struct iterator {
    using iterator_concept = std::input_iterator_tag;
    using value_type       = recursive_range;
    using difference_type  = std::ptrdiff_t;
    using reference        = recursive_range;

    reference operator*() const;

    iterator& operator++();
    iterator operator++(int);

    friend bool operator==(const iterator&, const iterator&);
  };

  iterator begin();
  iterator end();
};

static_assert(std::ranges::input_range<recursive_range>, "format_kind requires an input range");
static_assert(std::format_kind<recursive_range> == std::range_format::disabled);

static_assert(std::format_kind<std::filesystem::path> == std::range_format::disabled);

static_assert(std::format_kind<std::map<int, int>> == std::range_format::map);
static_assert(std::format_kind<std::multimap<int, int>> == std::range_format::map);
static_assert(std::format_kind<std::unordered_map<int, int>> == std::range_format::map);
static_assert(std::format_kind<std::unordered_multimap<int, int>> == std::range_format::map);

static_assert(std::format_kind<std::set<int>> == std::range_format::set);
static_assert(std::format_kind<std::multiset<int>> == std::range_format::set);
static_assert(std::format_kind<std::unordered_set<int>> == std::range_format::set);
static_assert(std::format_kind<std::unordered_multiset<int>> == std::range_format::set);

static_assert(std::format_kind<std::array<int, 1>> == std::range_format::sequence);
static_assert(std::format_kind<std::vector<int>> == std::range_format::sequence);
static_assert(std::format_kind<std::deque<int>> == std::range_format::sequence);
static_assert(std::format_kind<std::forward_list<int>> == std::range_format::sequence);
static_assert(std::format_kind<std::list<int>> == std::range_format::sequence);

static_assert(std::format_kind<std::span<int>> == std::range_format::sequence);

static_assert(std::format_kind<std::valarray<int>> == std::range_format::sequence);

// [format.range.fmtkind]/3
//   Remarks: Pursuant to [namespace.std], users may specialize format_kind for
//   cv-unqualified program-defined types that model ranges::input_range. Such
//   specializations shall be usable in constant expressions ([expr.const]) and
//   have type const range_format.
// Note only test the specializing, not all constraints.
struct no_specialization : std::ranges::view_base {
  using key_type = void;
  int* begin() const;
  int* end() const;
};
static_assert(std::format_kind<no_specialization> == std::range_format::set);

// The struct's "contents" are the same as no_specialization.
struct specialized : std::ranges::view_base {
  using key_type = void;
  int* begin() const;
  int* end() const;
};

template <>
constexpr std::range_format std::format_kind<specialized> = std::range_format::sequence;
static_assert(std::format_kind<specialized> == std::range_format::sequence);
