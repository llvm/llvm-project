//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// This test uses iterator types from std::filesystem, which were introduced in macOS 10.15.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// template<class I>
// concept __iterator_traits_detail::__cpp17_iterator;

#include <iterator>

#include <array>
#include <deque>
#ifndef _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY
#include <filesystem>
#endif
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iterator_traits_cpp17_iterators.h"

static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_proxy_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_input_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_proxy_input_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_forward_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_bidirectional_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<iterator_traits_cpp17_random_access_iterator>);

static_assert(std::__iterator_traits_detail::__cpp17_iterator<int*>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<int const*>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<int volatile*>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<int const volatile*>);

// <array>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::array<int, 10>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::array<int, 10>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::array<int, 10>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::array<int, 10>::const_reverse_iterator>);

// <deque>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::deque<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::deque<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::deque<int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::deque<int>::const_reverse_iterator>);

// <filesystem>
#ifndef _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::filesystem::directory_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::filesystem::recursive_directory_iterator>);
#endif

// <forward_list>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::forward_list<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::forward_list<int>::const_iterator>);

// <iterator>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::back_insert_iterator<std::vector<int>>>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::front_insert_iterator<std::vector<int>>>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::insert_iterator<std::vector<int>>>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::move_iterator<int*>>);

// <list>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::list<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::list<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::list<int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::list<int>::const_reverse_iterator>);

// <map>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::map<int, int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::map<int, int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::map<int, int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::map<int, int>::const_reverse_iterator>);

static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multimap<int, int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multimap<int, int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multimap<int, int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multimap<int, int>::const_reverse_iterator>);

// <set>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::set<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::set<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::set<int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::set<int>::const_reverse_iterator>);

static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multiset<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multiset<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multiset<int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::multiset<int>::const_reverse_iterator>);

// <string>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string::const_reverse_iterator>);

// <string_view>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string_view::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string_view::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string_view::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::string_view::const_reverse_iterator>);

// <unordered_map>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_map<int, int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_map<int, int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_map<int, int>::local_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_map<int, int>::const_local_iterator>);

static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multimap<int, int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multimap<int, int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multimap<int, int>::local_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multimap<int, int>::const_local_iterator>);

// <unordered_set>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_set<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_set<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_set<int>::local_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_set<int>::const_local_iterator>);

static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multiset<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multiset<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multiset<int>::local_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::unordered_multiset<int>::const_local_iterator>);

// <vector>
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::const_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::reverse_iterator>);
static_assert(std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::const_reverse_iterator>);

// Not iterators
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<void>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<void*>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int* const>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::iterator volatile>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::iterator&>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<std::vector<int>::iterator&&>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int[]>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int[10]>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int()>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int (*)()>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int (&)()>);

struct S {};
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<S>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int S::*>);
static_assert(!std::__iterator_traits_detail::__cpp17_iterator<int (S::*)()>);
