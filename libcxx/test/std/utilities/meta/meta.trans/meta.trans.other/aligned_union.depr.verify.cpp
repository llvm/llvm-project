//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <type_traits>

std::aligned_union<1, int>::type s2; // expected-warning {{'aligned_union<1, int>' is deprecated}}
std::aligned_union<3, char, char>::type s3; // expected-warning {{'aligned_union<3, char, char>' is deprecated}}
