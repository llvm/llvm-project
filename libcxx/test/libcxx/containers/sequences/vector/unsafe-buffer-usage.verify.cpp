//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc

// Make sure that std::vector's operations produce unsafe buffer access warnings when
// -Wunsafe-buffer-usage is used, when hardening is disabled.
//
// Note: We disable _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER to ensure that the libc++
//       headers are considered system headers, to validate that users would get
//       those diagnostics.
//
// ADDITIONAL_COMPILE_FLAGS: -Wunsafe-buffer-usage -U_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
// REQUIRES: libcpp-hardening-mode=none

#include <vector>
#include <cstddef>

void f(std::vector<int> v, std::vector<int> const cv, std::size_t n) {
  auto it = v.begin();

  (void)v[n];            // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)cv[n];           // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)v.front();       // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)cv.front();      // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)v.back();        // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)cv.back();       // expected-warning {{function introduces unsafe buffer manipulation}}
  v.pop_back();          // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)v.erase(it);     // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)v.erase(it, it); // expected-warning {{function introduces unsafe buffer manipulation}}

#if defined(_LIBCPP_ABI_BOUNDED_ITERATORS_IN_VECTOR)
  (void)*it;   // nothing
  (void)it[n]; // nothing
#else
  (void)*it;   // expected-warning {{function introduces unsafe buffer manipulation}}
  (void)it[n]; // expected-warning {{function introduces unsafe buffer manipulation}}
#endif
}
