//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: no-wide-characters

// <codecvt>

// Ensure that codecvt content is marked as removed.

#include <codecvt>

std::codecvt_mode c1;                // expected-error {{no type named 'codecvt_mode' in namespace 'std'}}
std::codecvt_utf8<wchar_t> c2;       // expected-error {{no template named 'codecvt_utf8' in namespace 'std'}}
std::codecvt_utf16<wchar_t> c3;      // expected-error {{no template named 'codecvt_utf16' in namespace 'std'}}
std::codecvt_utf8_utf16<wchar_t> c4; // expected-error {{no template named 'codecvt_utf8_utf16' in namespace 'std'}}
