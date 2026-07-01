//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

void containers() {
  std::deque<int> a;                   // expected-warning {{unused variable}}
  std::forward_list<int> b;            // expected-warning {{unused variable}}
  std::list<int> c;                    // expected-warning {{unused variable}}
  std::map<int, int> d;                // expected-warning {{unused variable}}
  std::multimap<int, int> e;           // expected-warning {{unused variable}}
  std::set<int> f;                     // expected-warning {{unused variable}}
  std::multiset<int> g;                // expected-warning {{unused variable}}
  std::unordered_map<int, int> h;      // expected-warning {{unused variable}}
  std::unordered_multimap<int, int> i; // expected-warning {{unused variable}}
  std::unordered_set<int> j;           // expected-warning {{unused variable}}
  std::unordered_multiset<int> k;      // expected-warning {{unused variable}}
  std::string l;                       // expected-warning {{unused variable}}
  std::vector<int> m;                  // expected-warning {{unused variable}}
  std::vector<bool> n;                 // expected-warning {{unused variable}}
}

void container_iterators() {
  std::deque<int>::iterator a;                   // expected-warning {{unused variable}}
  std::forward_list<int>::iterator b;            // expected-warning {{unused variable}}
  std::list<int>::iterator c;                    // expected-warning {{unused variable}}
  std::map<int, int>::iterator d;                // expected-warning {{unused variable}}
  std::multimap<int, int>::iterator e;           // expected-warning {{unused variable}}
  std::set<int>::iterator f;                     // expected-warning {{unused variable}}
  std::multiset<int>::iterator g;                // expected-warning {{unused variable}}
  std::unordered_map<int, int>::iterator h;      // expected-warning {{unused variable}}
  std::unordered_multimap<int, int>::iterator i; // expected-warning {{unused variable}}
  std::unordered_set<int>::iterator j;           // expected-warning {{unused variable}}
  std::unordered_multiset<int>::iterator k;      // expected-warning {{unused variable}}
  std::string::iterator l;                       // expected-warning {{unused variable}}
  std::vector<int>::iterator m;                  // expected-warning {{unused variable}}
  std::vector<bool>::iterator n;                 // expected-warning {{unused variable}}
}

void container_const_iterators() {
  std::deque<int>::const_iterator a;                   // expected-warning {{unused variable}}
  std::forward_list<int>::const_iterator b;            // expected-warning {{unused variable}}
  std::list<int>::const_iterator c;                    // expected-warning {{unused variable}}
  std::map<int, int>::const_iterator d;                // expected-warning {{unused variable}}
  std::multimap<int, int>::const_iterator e;           // expected-warning {{unused variable}}
  std::set<int>::const_iterator f;                     // expected-warning {{unused variable}}
  std::multiset<int>::const_iterator g;                // expected-warning {{unused variable}}
  std::unordered_map<int, int>::const_iterator h;      // expected-warning {{unused variable}}
  std::unordered_multimap<int, int>::const_iterator i; // expected-warning {{unused variable}}
  std::unordered_set<int>::const_iterator j;           // expected-warning {{unused variable}}
  std::unordered_multiset<int>::const_iterator k;      // expected-warning {{unused variable}}
  std::string::const_iterator l;                       // expected-warning {{unused variable}}
  std::vector<int>::const_iterator m;                  // expected-warning {{unused variable}}
  std::vector<bool>::const_iterator n;                 // expected-warning {{unused variable}}
}
