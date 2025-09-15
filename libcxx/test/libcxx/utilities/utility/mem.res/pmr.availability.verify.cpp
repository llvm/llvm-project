//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: availability-pmr-missing

// TODO: This test doesn't work until https://llvm.org/PR40995
//       has been fixed, because we actually disable availability markup.
// XFAIL: *

// Test the availability markup on std::pmr components.

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory_resource>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

void f() {
    [[maybe_unused]] std::pmr::match_results<const char8_t*> m1; // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::cmatch m2;                        // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::wcmatch m3;                       // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::smatch m4;                        // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::wsmatch m5;                       // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::deque<int> m6;                    // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::forward_list<int> m7;             // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::list<int> m8;                     // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::map<int, int> m9;                 // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::multimap<int, int> m10;           // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::set<int> m11;                     // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::multiset<int> m12;                // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::string m13;                       // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::wstring m14;                      // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::u8string m15;                     // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::u16string m16;                    // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::u32string m17;                    // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::basic_string<char8_t> m18;        // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::unordered_map<int, int> m19;      // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::unordered_multimap<int, int> m20; // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::unordered_set<int, int> m21;      // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::unordered_multiset<int, int> m22; // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::vector<int> m23;                  // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::polymorphic_allocator<int> poly;  // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::memory_resource* res = nullptr;   // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::synchronized_pool_resource r1;    // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::monotonic_buffer_resource r2;     // expected-error {{is unavailable}}
    [[maybe_unused]] std::pmr::unsynchronized_pool_resource r3;  // expected-error {{is unavailable}}
    (void)std::pmr::get_default_resource();                      // expected-error {{is unavailable}}
    (void)std::pmr::set_default_resource(nullptr);               // expected-error {{is unavailable}}
    (void)std::pmr::new_delete_resource();                       // expected-error {{is unavailable}}
    (void)std::pmr::null_memory_resource();                      // expected-error {{is unavailable}}
    (void)(*res == *res);                                        // expected-error {{is unavailable}}
}
