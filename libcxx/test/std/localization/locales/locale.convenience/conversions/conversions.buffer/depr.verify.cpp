//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: no-wide-characters

// <codecvt>

// ensure that wbuffer_convert is marked as deprecated

#include <codecvt>
#include <locale>

std::wbuffer_convert<std::codecvt_utf8<wchar_t>> c1; // expected-warning {{'wbuffer_convert<std::codecvt_utf8<wchar_t, 1114111, 0>>' is deprecated}}
// expected-warning@-1 {{'codecvt_utf8<wchar_t, 1114111, 0>' is deprecated}}
