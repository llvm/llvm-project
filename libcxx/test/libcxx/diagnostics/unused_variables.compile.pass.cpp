//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we don't introduce new warnings on unused variables of libc++ classes if
// _LIBCPP_DISABLE_UNUSED_STRUCT_WARNINGS is set.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_UNUSED_STRUCT_WARNINGS

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

void containers() {
  std::deque<int> a;
  std::forward_list<int> b;
  std::list<int> c;
  std::map<int, int> d;
  std::multimap<int, int> e;
  std::set<int> f;
  std::multiset<int> g;
  std::unordered_map<int, int> h;
  std::unordered_multimap<int, int> i;
  std::unordered_set<int> j;
  std::unordered_multiset<int> k;
  std::string l;
  std::vector<int> m;
  std::vector<bool> n;
}

void container_iterators() {
  std::deque<int>::iterator a;
#if TEST_STD_VER <= 23
  std::forward_list<int>::iterator b;
  std::list<int>::iterator c;
  std::map<int, int>::iterator d;
  std::multimap<int, int>::iterator e;
  std::set<int>::iterator f;
  std::multiset<int>::iterator g;
#endif
  std::unordered_map<int, int>::iterator h;
  std::unordered_multimap<int, int>::iterator i;
  std::unordered_set<int>::iterator j;
  std::unordered_multiset<int>::iterator k;
#if TEST_STD_VER <= 11
  std::string::iterator l;
  std::vector<int>::iterator m;
#endif
#if TEST_STD_VER <= 17
  std::vector<bool>::iterator n;
#endif
}

void container_const_iterators() {
  std::deque<int>::const_iterator a;
#if TEST_STD_VER <= 23
  std::forward_list<int>::const_iterator b;
  std::list<int>::const_iterator c;
  std::map<int, int>::const_iterator d;
  std::multimap<int, int>::const_iterator e;
  std::set<int>::const_iterator f;
  std::multiset<int>::const_iterator g;
#endif
  std::unordered_map<int, int>::const_iterator h;
  std::unordered_multimap<int, int>::const_iterator i;
  std::unordered_set<int>::const_iterator j;
  std::unordered_multiset<int>::const_iterator k;
#if TEST_STD_VER <= 11
  std::string::const_iterator l;
  std::vector<int>::const_iterator m;
#endif
#if TEST_STD_VER <= 17
  std::vector<bool>::const_iterator n;
#endif
}
