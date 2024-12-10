//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc

// Make sure that std::string's operations produce unsafe buffer access warnings when
// -Wunsafe-buffer-usage is used, when hardening is disabled.
//
// Note: We disable _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER to ensure that the libc++
//       headers are considered system headers, to validate that users would get
//       those diagnostics.
//
// ADDITIONAL_COMPILE_FLAGS: -Wunsafe-buffer-usage -U_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
// REQUIRES: libcpp-hardening-mode=none

#include <string>
#include <cstddef>

void f(std::string s, std::string const cs, std::size_t n) {
  (void)s[n];       // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)cs[n];      // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.front();  // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)cs.front(); // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)s.back();   // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)cs.back();  // expected-warning {{function introduces unsafe buffer manipulation}}
  s.pop_back();     // expected-warning {{function introduces unsafe buffer manipulation}}

  auto it = s.begin();
#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STRING)
  (void)*it;   // nothing
  (void)it[n]; // nothing
#else
  (void)*it;   // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)it[n]; // expected-warning {{function introduces unsafe buffer manipulation}}
#endif
}
