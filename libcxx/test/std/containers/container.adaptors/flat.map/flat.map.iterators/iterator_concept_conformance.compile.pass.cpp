//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <flat_map>
#include <deque>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "test_macros.h"

void test() {
  {
    using C   = std::flat_map<int, char>;
    using I   = C::iterator;
    using CI  = C::const_iterator;
    using RI  = C::reverse_iterator;
    using CRI = C::const_reverse_iterator;
    static_assert(std::random_access_iterator<I>);
    static_assert(std::random_access_iterator<CI>);
    static_assert(std::random_access_iterator<RI>);
    static_assert(std::random_access_iterator<CRI>);
    static_assert(!std::contiguous_iterator<I>);
    static_assert(!std::contiguous_iterator<CI>);
    static_assert(!std::contiguous_iterator<RI>);
    static_assert(!std::contiguous_iterator<CRI>);
    static_assert(!std::indirectly_writable<I, std::pair<int, char>>);
    static_assert(!std::indirectly_writable<CI, std::pair<int, char>>);
    static_assert(!std::indirectly_writable<RI, std::pair<int, char>>);
    static_assert(!std::indirectly_writable<CRI, std::pair<int, char>>);
    static_assert(std::sentinel_for<I, I>);
    static_assert(std::sentinel_for<I, CI>);
    static_assert(!std::sentinel_for<I, RI>);
    static_assert(!std::sentinel_for<I, CRI>);
    static_assert(std::sentinel_for<CI, I>);
    static_assert(std::sentinel_for<CI, CI>);
    static_assert(!std::sentinel_for<CI, RI>);
    static_assert(!std::sentinel_for<CI, CRI>);
    static_assert(!std::sentinel_for<RI, I>);
    static_assert(!std::sentinel_for<RI, CI>);
    static_assert(std::sentinel_for<RI, RI>);
    static_assert(std::sentinel_for<RI, CRI>);
    static_assert(!std::sentinel_for<CRI, I>);
    static_assert(!std::sentinel_for<CRI, CI>);
    static_assert(std::sentinel_for<CRI, RI>);
    static_assert(std::sentinel_for<CRI, CRI>);
    static_assert(std::indirectly_movable_storable<I, std::pair<int, char>*>);
    static_assert(std::indirectly_movable_storable<CI, std::pair<int, char>*>);
    static_assert(std::indirectly_movable_storable<RI, std::pair<int, char>*>);
    static_assert(std::indirectly_movable_storable<CRI, std::pair<int, char>*>);
  }
  {
    using C   = std::flat_map<char*, int, std::less<>, std::deque<char*>, std::vector<int>>;
    using I   = C::iterator;
    using CI  = C::const_iterator;
    using RI  = C::reverse_iterator;
    using CRI = C::const_reverse_iterator;
    static_assert(std::random_access_iterator<I>);
    static_assert(std::random_access_iterator<CI>);
    static_assert(std::random_access_iterator<RI>);
    static_assert(std::random_access_iterator<CRI>);
    static_assert(!std::contiguous_iterator<I>);
    static_assert(!std::contiguous_iterator<CI>);
    static_assert(!std::contiguous_iterator<RI>);
    static_assert(!std::contiguous_iterator<CRI>);
    static_assert(!std::indirectly_writable<I, std::pair<char*, int>>);
    static_assert(!std::indirectly_writable<CI, std::pair<char*, int>>);
    static_assert(!std::indirectly_writable<RI, std::pair<char*, int>>);
    static_assert(!std::indirectly_writable<CRI, std::pair<char*, int>>);
    static_assert(std::sentinel_for<I, I>);
    static_assert(std::sentinel_for<I, CI>);
    static_assert(!std::sentinel_for<I, RI>);
    static_assert(!std::sentinel_for<I, CRI>);
    static_assert(std::sentinel_for<CI, I>);
    static_assert(std::sentinel_for<CI, CI>);
    static_assert(!std::sentinel_for<CI, RI>);
    static_assert(!std::sentinel_for<CI, CRI>);
    static_assert(!std::sentinel_for<RI, I>);
    static_assert(!std::sentinel_for<RI, CI>);
    static_assert(std::sentinel_for<RI, RI>);
    static_assert(std::sentinel_for<RI, CRI>);
    static_assert(!std::sentinel_for<CRI, I>);
    static_assert(!std::sentinel_for<CRI, CI>);
    static_assert(std::sentinel_for<CRI, RI>);
    static_assert(std::sentinel_for<CRI, CRI>);
    static_assert(std::indirectly_movable_storable<I, std::pair<char*, int>*>);
    static_assert(std::indirectly_movable_storable<CI, std::pair<char*, int>*>);
    static_assert(std::indirectly_movable_storable<RI, std::pair<char*, int>*>);
    static_assert(std::indirectly_movable_storable<CRI, std::pair<char*, int>*>);
  }
  {
    using C   = std::flat_map<char, bool, std::less<>, std::string, std::vector<bool>>;
    using I   = C::iterator;
    using CI  = C::const_iterator;
    using RI  = C::reverse_iterator;
    using CRI = C::const_reverse_iterator;
    static_assert(std::random_access_iterator<I>);
    static_assert(std::random_access_iterator<CI>);
    static_assert(std::random_access_iterator<RI>);
    static_assert(std::random_access_iterator<CRI>);
    static_assert(!std::contiguous_iterator<I>);
    static_assert(!std::contiguous_iterator<CI>);
    static_assert(!std::contiguous_iterator<RI>);
    static_assert(!std::contiguous_iterator<CRI>);
    static_assert(!std::indirectly_writable<I, std::pair<char, bool>>);
    static_assert(!std::indirectly_writable<CI, std::pair<char, bool>>);
    static_assert(!std::indirectly_writable<RI, std::pair<char, bool>>);
    static_assert(!std::indirectly_writable<CRI, std::pair<char, bool>>);
    static_assert(std::sentinel_for<I, I>);
    static_assert(std::sentinel_for<I, CI>);
    static_assert(!std::sentinel_for<I, RI>);
    static_assert(!std::sentinel_for<I, CRI>);
    static_assert(std::sentinel_for<CI, I>);
    static_assert(std::sentinel_for<CI, CI>);
    static_assert(!std::sentinel_for<CI, RI>);
    static_assert(!std::sentinel_for<CI, CRI>);
    static_assert(!std::sentinel_for<RI, I>);
    static_assert(!std::sentinel_for<RI, CI>);
    static_assert(std::sentinel_for<RI, RI>);
    static_assert(std::sentinel_for<RI, CRI>);
    static_assert(!std::sentinel_for<CRI, I>);
    static_assert(!std::sentinel_for<CRI, CI>);
    static_assert(std::sentinel_for<CRI, RI>);
    static_assert(std::sentinel_for<CRI, CRI>);
    static_assert(std::indirectly_movable_storable<I, std::pair<char, bool>*>);
    static_assert(std::indirectly_movable_storable<CI, std::pair<char, bool>*>);
    static_assert(std::indirectly_movable_storable<RI, std::pair<char, bool>*>);
    static_assert(std::indirectly_movable_storable<CRI, std::pair<char, bool>*>);
  }
}
