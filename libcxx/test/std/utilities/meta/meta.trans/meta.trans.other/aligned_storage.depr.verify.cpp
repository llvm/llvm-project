//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <type_traits>

std::aligned_storage<1, 1>::type s2; // expected-warning {{'aligned_storage<1, 1>' is deprecated}}
std::aligned_storage<3, 1>::type s3; // expected-warning {{'aligned_storage<3, 1>' is deprecated}}
