//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <any>
#include <barrier>
#include <condition_variable>
#include <deque>
#include <exception>
#include <expected>
#include <filesystem>
#include <flat_map>
#include <flat_set>
#include <format>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

template <class T>
T get();

void containers() {
  std::deque<int> a;                   // expected-warning {{unused variable}}
  std::deque<int>::iterator b;         // expected-warning {{unused variable}}
  std::forward_list<int> c;            // expected-warning {{unused variable}}
  std::list<int> d;                    // expected-warning {{unused variable}}
  std::map<int, int> e;                // expected-warning {{unused variable}}
  std::multimap<int, int> f;           // expected-warning {{unused variable}}
  std::set<int> g;                     // expected-warning {{unused variable}}
  std::multiset<int> h;                // expected-warning {{unused variable}}
  std::unordered_map<int, int> i;      // expected-warning {{unused variable}}
  std::unordered_multimap<int, int> j; // expected-warning {{unused variable}}
  std::unordered_set<int> k;           // expected-warning {{unused variable}}
  std::unordered_multiset<int> l;      // expected-warning {{unused variable}}
  std::string m;                       // expected-warning {{unused variable}}
  std::vector<int> n;                  // expected-warning {{unused variable}}
  std::vector<bool> o;                 // expected-warning {{unused variable}}
}

void container_adaptors() {
  std::flat_map<int, int> a;      // expected-warning {{unused variable}}
  std::flat_multimap<int, int> b; // expected-warning {{unused variable}}
  std::flat_set<int> c;           // expected-warning {{unused variable}}
  std::flat_multiset<int> d;      // expected-warning {{unused variable}}
}

void expected() {
  std::bad_expected_access<int> a(0); // expected-warning {{unused variable}}
  std::expected<void, std::any> b;    // expected-warning {{unused variable}}
  std::expected<int, std::any> c;     // expected-warning {{unused variable}}
  std::unexpected<std::any> d(1);     // expected-warning {{unused variable}}
}

void filesystem() {
  fs::directory_entry a;              // expected-warning {{unused variable}}
  fs::directory_iterator b;           // expected-warning {{unused variable}}
  fs::file_status c;                  // expected-warning {{unused variable}}
  fs::filesystem_error d("", {});     // expected-warning {{unused variable}}
  fs::path e;                         // expected-warning {{unused variable}}
  fs::path::iterator f;               // expected-warning {{unused variable}}
  fs::recursive_directory_iterator g; // expected-warning {{unused variable}}
}

void format() {
  std::basic_format_arg<std::format_context> a; // expected-warning {{unused variable}}
  auto b = get<std::format_context>();          // expected-warning {{unused variable}}
  std::format_error c("");                      // expected-warning {{unused variable}}
}

void future() {
  std::future<void> a;          // expected-warning {{unused variable}}
  std::future<int&> b;          // expected-warning {{unused variable}}
  std::future<int> c;           // expected-warning {{unused variable}}
  std::promise<void> d;         // expected-warning {{unused variable}}
  std::promise<int&> e;         // expected-warning {{unused variable}}
  std::promise<int> f;          // expected-warning {{unused variable}}
  std::packaged_task<void()> g; // expected-warning {{unused variable}}
  std::packaged_task<int()> h;  // expected-warning {{unused variable}}
  std::shared_future<void> i;   // expected-warning {{unused variable}}
  std::shared_future<int&> j;   // expected-warning {{unused variable}}
  std::shared_future<int> k;    // expected-warning {{unused variable}}
}

void generic_exceptions() {
  std::exception a;        // expected-warning {{unused variable}}
  std::bad_exception b;    // expected-warning {{unused variable}}
  std::exception_ptr c;    // expected-warning {{unused variable}}
  std::nested_exception d; // expected-warning {{unused variable}}
}

void iterator() {
  using C = std::deque<int>;
  C container;

  std::back_insert_iterator<C> a(container);               // expected-warning {{unused variable}}
  std::front_insert_iterator<C> b(container);              // expected-warning {{unused variable}}
  std::insert_iterator<C> c(container, container.begin()); // expected-warning {{unused variable}}
  std::istream_iterator<char> d    = std::cin;             // expected-warning {{unused variable}}
  std::istreambuf_iterator<char> e = std::cin;             // expected-warning {{unused variable}}
  std::move_iterator<C::iterator> f;                       // expected-warning {{unused variable}}
}

void streams() {
  std::ifstream a; // expected-warning {{unused variable}}
  std::ofstream b; // expected-warning {{unused variable}}
  std::fstream c;  // expected-warning {{unused variable}}
  std::filebuf d;  // expected-warning {{unused variable}}
  std::filebuf buf;
  std::istream e(&buf);  // expected-warning {{unused variable}}
  std::ostream f(&buf);  // expected-warning {{unused variable}}
  std::iostream g(&buf); // expected-warning {{unused variable}}
}

void synchronization() {
  std::barrier<> a(0);           // expected-warning {{unused variable}}
  std::condition_variable b;     // expected-warning {{unused variable}}
  std::condition_variable_any c; // expected-warning {{unused variable}}
}

void other() {
  std::any a; // expected-warning {{unused variable}}
  int bi;
  std::reference_wrapper<int> b(bi); // expected-warning {{unused variable}}
  std::bad_function_call c;          // expected-warning {{unused variable}}
  std::function<void()> d;           // expected-warning {{unused variable}}
}
