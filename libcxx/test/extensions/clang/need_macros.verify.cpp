//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// gcc supports "partial inclusion" of some builtin headers via __need macros.
// These are implemented in modules via textual headers, which are tricky to
// support because libc++ needs to know about them and make its corresponding
// headers textual too so that the macros are passed along.
//
// Another wrinkle is that __need macros can be used to force a type to be
// when it wouldn't normally be. e.g. Apple platforms will use __need_rsize_t
// to force the declaration of rsize_t to be visible even when
// __STDC_WANT_LIB_EXT1__ isn't set.

// gcc doesn't support all of the __need macros
// UNSUPPORTED: gcc

// The frozen C++03 headers don't support the __need macros tested here.
// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// Some of the __need macros are new in clang 22.x and Apple clang 21.x (there
// isn't an Apple clang 18-20)
// UNSUPPORTED: clang-20, clang-21, apple-clang-17

// float.h doesn't always define INFINITY and NAN.
#define __need_infinity_nan
#include <float.h>
constexpr float infinity = INFINITY;
constexpr float nan      = NAN;

// Make sure that only the __need'ed interfaces are availbale. Note that the
// C++ headers like <cstdarg> do -not- support partial inclusion, only the
// "compatibility" headers.
#define __need_va_list
#include <stdarg.h>
static void func(int param, ...) {
  va_list args;
  va_start(args, param); // expected-error {{use of undeclared identifier 'va_start'}}
}

// stddef.h doesn't always define max_align_t.
#define __need_max_align_t
#define __need_size_t
#include <stddef.h>
static_assert(alignof(max_align_t) >= alignof(long double), "");
