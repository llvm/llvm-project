//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we warn on unused variables of libc++ classes which behave like value types.
// ADDITIONAL_COMPILE_FLAGS: -Wunused-variable

#include <barrier>
#include <condition_variable>
#include <deque>
#include <forward_list>
#include <latch>
#include <list>
#include <map>
#include <mutex>
#include <semaphore>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test_macros.h"

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

#ifndef TEST_HAS_NO_THREADS
void synchronization(std::mutex& m) {
  // <mutex>
  std::mutex a;                      // expected-warning {{unused variable}}
  std::once_flag b;                  // expected-warning {{unused variable}}
  std::recursive_mutex c;            // expected-warning {{unused variable}}
  std::recursive_timed_mutex d;      // expected-warning {{unused variable}}
  std::timed_mutex e;                // expected-warning {{unused variable}}
  std::unique_lock<std::mutex> f;    // TODO: We should warn on this
  std::unique_lock<std::mutex> g(m); // Shouldn't be diagnosed
#  if TEST_STD_VER >= 17
  std::unique_lock<std::mutex> h(m, std::defer_lock); // TODO: We should warn on this
#  endif

  // <condition_variable>
  std::condition_variable i;     // expected-warning {{unused variable}}
  std::condition_variable_any j; // expected-warning {{unused variable}}

#  if TEST_STD_VER >= 20
  // <semaphore>
  std::counting_semaphore<> k(1); // expected-warning {{unused variable}}
#  endif

  // <shared_mutex>
#  if TEST_STD_VER >= 17
  std::shared_mutex l; // expected-warning {{unused variable}}
#  endif
#  if TEST_STD_VER >= 14
  std::shared_timed_mutex n; // expected-warning {{unused variable}}
  std::shared_timed_mutex n2;
  std::shared_lock<std::shared_timed_mutex> o;                      // TODO: We should warn on this
  std::shared_lock<std::shared_timed_mutex> p(n2);                  // Shouldn't be diagnosed
  std::shared_lock<std::shared_timed_mutex> q(n2, std::defer_lock); // TODO: We should warn on this
#  endif

#  if TEST_STD_VER >= 20
  // <barrier>
  std::barrier<> r(1); // expected-warning {{unused variable}}

  // <latch>
  std::latch s(1); // expected-warning {{unused variable}}
#  endif
}
#endif // TEST_HAS_NO_THREADS
