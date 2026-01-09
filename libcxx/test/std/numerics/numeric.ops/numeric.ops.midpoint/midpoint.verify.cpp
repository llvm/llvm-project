//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: std-at-least-c++20

// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept

// An overload exists for each of char and all arithmetic types except bool.

#include <numeric>

#include "test_macros.h"

int func1() { return 1; }
int func2() { return 2; }

struct Incomplete;
Incomplete* ip = nullptr;
void* vp       = nullptr;

void test() {
  // expected-error@+1 {{no matching function for call to 'midpoint'}}
  (void)std::midpoint(false, true);
  // expected-error@+1 {{no matching function for call to 'midpoint'}}
  (void)std::midpoint<const bool>(false, true);
  // expected-error@+1 {{no matching function for call to 'midpoint'}}
  (void)std::midpoint<const volatile bool>(false, true);
  // expected-error@+1 {{no matching function for call to 'midpoint'}}
  (void)std::midpoint<volatile bool>(false, true);

  //  A couple of odd pointer types that should fail
  (void)std::midpoint(nullptr, nullptr); // expected-error {{no matching function for call to 'midpoint'}}
  (void)std::midpoint(func1, func2);     // expected-error {{no matching function for call to 'midpoint'}}
  (void)std::midpoint(ip, ip);           // expected-error {{no matching function for call to 'midpoint'}}
  (void)std::midpoint(vp, vp);           // expected-error {{no matching function for call to 'midpoint'}}
}
