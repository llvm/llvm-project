//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make use that `__specialized_algorithm::__has_algorithm` is true when we expect it to be

// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -Wno-c++14-extensions -Wno-c++17-extensions

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "test_macros.h"

template <class... Args>
inline constexpr bool has_alg = std::__specialized_algorithm<Args...>::__has_algorithm;

template <class T>
using single_iter = std::__single_iterator<T>;

template <class T, class U>
using iter_pair = std::__iterator_pair<T, U>;

template <class T>
using single_range = std::__single_range<T>;

namespace Alg = std::_Algorithm;

namespace vector_bool {
using iter       = typename std::vector<bool>::iterator;
using const_iter = typename std::vector<bool>::const_iterator;
static_assert(has_alg<Alg::__fill_n, single_iter<iter>>);
static_assert(has_alg<Alg::__copy, iter_pair<iter, iter>, single_iter<iter>>);
static_assert(has_alg<Alg::__copy, iter_pair<const_iter, const_iter>, single_iter<iter>>);
static_assert(has_alg<Alg::__swap_ranges, iter_pair<iter, iter>, single_iter<iter>>);
} // namespace vector_bool

namespace set {
using iter       = typename std::set<int>::iterator;
using const_iter = typename std::set<int>::const_iterator;
static_assert(has_alg<Alg::__for_each, iter_pair<iter, iter>>);
static_assert(has_alg<Alg::__for_each, iter_pair<const_iter, const_iter>>);
#if TEST_STD_VER >= 14
static_assert(has_alg<Alg::__for_each, single_range<std::set<int>>>);
#endif
} // namespace set

namespace multiset {
using iter       = typename std::multiset<int>::iterator;
using const_iter = typename std::multiset<int>::const_iterator;
static_assert(has_alg<Alg::__for_each, iter_pair<iter, iter>>);
static_assert(has_alg<Alg::__for_each, iter_pair<const_iter, const_iter>>);
#if TEST_STD_VER >= 14
static_assert(has_alg<Alg::__for_each, single_range<std::multiset<int>>>);
#endif
} // namespace multiset

namespace map {
using iter       = typename std::map<int, int>::iterator;
using const_iter = typename std::map<int, int>::const_iterator;
static_assert(has_alg<Alg::__for_each, iter_pair<iter, iter>>);
static_assert(has_alg<Alg::__for_each, iter_pair<const_iter, const_iter>>);
#if TEST_STD_VER >= 14
static_assert(has_alg<Alg::__for_each, single_range<std::map<int, int>>>);
#endif
} // namespace map

namespace multimap {
using iter       = typename std::multimap<int, int>::iterator;
using const_iter = typename std::multimap<int, int>::const_iterator;
static_assert(has_alg<Alg::__for_each, iter_pair<iter, iter>>);
static_assert(has_alg<Alg::__for_each, iter_pair<const_iter, const_iter>>);
#if TEST_STD_VER >= 14
static_assert(has_alg<Alg::__for_each, single_range<std::multimap<int, int>>>);
#endif
} // namespace multimap

namespace ostreambuf_iterator {
template <class CharT>
using obi = std::ostreambuf_iterator<CharT>;
static_assert(has_alg<Alg::__copy, iter_pair<char*, char*>, single_iter<obi<char>>>);
static_assert(has_alg<Alg::__copy, iter_pair<const char*, const char*>, single_iter<obi<char>>>);
static_assert(has_alg<Alg::__copy, iter_pair<wchar_t*, wchar_t*>, single_iter<obi<wchar_t>>>);
static_assert(has_alg<Alg::__copy, iter_pair<const wchar_t*, const wchar_t*>, single_iter<obi<wchar_t>>>);
static_assert(!has_alg<Alg::__copy, iter_pair<char*, char*>, single_iter<obi<wchar_t>>>);

} // namespace ostreambuf_iterator
