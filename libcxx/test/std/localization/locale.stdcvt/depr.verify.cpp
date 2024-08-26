//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++26
// UNSUPPORTED: no-wide-characters

// <codecvt>

// Ensure that codecvt content is marked as deprecated.
// The header has been removed in C++26.

#include <codecvt>

std::codecvt_mode c1; // expected-warning {{'codecvt_mode' is deprecated}}
std::codecvt_utf8<wchar_t> c2; // expected-warning-re {{'codecvt_utf8<wchar_t{{.*}}>' is deprecated}}
std::codecvt_utf16<wchar_t> c3; // expected-warning-re {{'codecvt_utf16<wchar_t{{.*}}>' is deprecated}}
std::codecvt_utf8_utf16<wchar_t> c4; // expected-warning-re {{'codecvt_utf8_utf16<wchar_t{{.*}}>' is deprecated}}
