//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc

// Make sure that std::span's operations produce unsafe buffer access warnings when
// -Wunsafe-buffer-usage is used, when hardening is disabled.
//
// Note: We disable _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER to ensure that the libc++
//       headers are considered system headers, to validate that users would get
//       those diagnostics.
//
// ADDITIONAL_COMPILE_FLAGS: -Wunsafe-buffer-usage -U_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
// REQUIRES: libcpp-hardening-mode=none

#include <span>
#include <cstddef>

void f1(std::span<int, std::dynamic_extent> s, std::size_t n) {
  (void)s.first<10>();       // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.first(n);          // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.last<10>();        // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.last(n);           // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.subspan<10, 20>(); // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.subspan<10>();     // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.subspan(n, n);     // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.subspan(n);        // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s[n];                // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.front();           // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.back();            // expected-warning {{function introduces unsafe buffer manipulation}}

  auto it = s.begin();
#ifdef _LIBCPP_ABI_BOUNDED_ITERATORS
  (void)*it;   // nothing
  (void)it[n]; // nothing
#else
  (void)*it;   // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)it[n]; // expected-warning {{function introduces unsafe buffer manipulation}}
#endif
}

void f2(std::span<int, 1024> s, std::size_t n) {
  (void)s.first<10>();       // nothing
  (void)s.first(n);          // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.last<10>();        // nothing
  (void)s.last(n);           // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.subspan<10, 20>(); // nothing
  (void)s.subspan(n, n);     // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.subspan(n);        // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s[n];                // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.front();           // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.back();            // expected-warning {{function introduces unsafe buffer manipulation}}

  auto it = s.begin();
#ifdef _LIBCPP_ABI_BOUNDED_ITERATORS
  (void)*it;   // nothing
  (void)it[n]; // nothing
#else
  (void)*it;   // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)it[n]; // expected-warning {{function introduces unsafe buffer manipulation}}
#endif
}
