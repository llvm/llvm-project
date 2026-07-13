//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <utility>

// template <typename T>
// [[nodiscard]] constexpr
// auto forward_like(auto&& x) noexcept -> see below;

// Mandates: T is a referenceable type (3.45 [defns.referenceable]).

#include <utility>

struct incomplete;

void test() {
  int i;
  (void)std::forward_like<incomplete>(i);

  (void)std::forward_like<void>(i);                // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<const void>(i);          // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<volatile void>(i);       // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<const volatile void>(i); // expected-error {{no matching function for call to 'forward_like'}}

  using fp   = void();
  using cfp  = void() const;
  using vfp  = void() volatile;
  using cvfp = void() const volatile;
  (void)std::forward_like<fp>(i);
  (void)std::forward_like<cfp>(i);  // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<cfp>(i);  // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<vfp>(i);  // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<cvfp>(i); // expected-error {{no matching function for call to 'forward_like'}}

  using fpr  = void()&;
  using fprr = void()&&;
  (void)std::forward_like<fpr>(i);  // expected-error {{no matching function for call to 'forward_like'}}
  (void)std::forward_like<fprr>(i); // expected-error {{no matching function for call to 'forward_like'}}
}
