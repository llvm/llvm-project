//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-filesystem-library

// std::filesystem is unavailable prior to macOS 10.15
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// Make sure the various containers' iterators are not broken by the use of `std::rel_ops`.

#include <utility> // for std::rel_ops

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test_macros.h"

#if TEST_STD_VER >= 11
#include "filesystem_include.h"
#endif

#if TEST_STD_VER >= 17
#include <string_view>
#endif

#if TEST_STD_VER >= 20
#include <span>
#endif

using namespace std::rel_ops;

template<class It, class ConstIt>
void test_eq(It it, ConstIt cit) {
    (void)(it == it);
    (void)(it != it);
    (void)(it == cit);
    (void)(it != cit);
    (void)(cit == it);
    (void)(cit != it);
    (void)(cit == cit);
    (void)(cit != cit);
}

template<class It, class ConstIt>
void test_lt(It it, ConstIt cit) {
    (void)(it <  it);
    (void)(it <= it);
    (void)(it >  it);
    (void)(it >= it);
    (void)(it <  cit);
    (void)(it <= cit);
    (void)(it >  cit);
    (void)(it >= cit);
    (void)(cit <  it);
    (void)(cit <= it);
    (void)(cit >  it);
    (void)(cit >= it);
    (void)(cit <  cit);
    (void)(cit <= cit);
    (void)(cit >  cit);
    (void)(cit >= cit);

    // Test subtraction too, even though std::rel_ops shouldn't affect it.

    (void)(it - it);
    (void)(it - cit);
    (void)(cit - it);
    (void)(cit - cit);
}

template<class Container>
void test_forward() {
    // There is no need to distinguish "forward" from "bidirectional."
    // libc++ already can't handle `c.rbegin() >= c.rbegin()` in the
    // presence of std::rel_ops, and neither can Microsoft nor libstdc++.

    Container c;
    typename Container::iterator it = c.begin();
    typename Container::const_iterator cit = c.begin();
    test_eq(it, cit);
}

template<class Container>
void test_random_access() {
    Container c;
    typename Container::iterator it = c.begin();
    typename Container::const_iterator cit = c.begin();
    test_eq(it, cit);
    test_lt(it, cit);
}

template void test_random_access<std::array<int, 10> >();
template void test_random_access<std::deque<int> >();
template void test_forward<std::forward_list<int> >();
template void test_forward<std::list<int> >();
template void test_forward<std::map<int, int> >();
template void test_forward<std::multimap<int, int> >();
template void test_forward<std::multiset<int> >();
template void test_forward<std::set<int> >();
template void test_random_access<std::string>();
template void test_forward<std::unordered_map<int, int> >();
template void test_forward<std::unordered_multimap<int, int> >();
template void test_forward<std::unordered_multiset<int> >();
template void test_forward<std::unordered_set<int> >();
template void test_random_access<std::vector<int> >();

#if TEST_STD_VER >= 11
void test_directory_iterators() {
    fs::directory_iterator it;
    test_eq(it, it);

    fs::recursive_directory_iterator rdit;
    test_eq(rdit, rdit);
}

template void test_forward<fs::path>();
#endif

#if TEST_STD_VER >= 17
template void test_random_access<std::string_view>();
#endif

#if TEST_STD_VER >= 20
void test_span() {
    std::span<int> c;
    std::span<int>::iterator it = c.begin();  // span has no const_iterator
    test_eq(it, it);
    test_lt(it, it);
}
#endif
