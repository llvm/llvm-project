//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-5, clang-6
// UNSUPPORTED: apple-clang-6, apple-clang-7, apple-clang-8, apple-clang-9, apple-clang-10

// <chrono>
// class year;

// constexpr year operator""y(unsigned long long y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using std::chrono::year;
    year d1 = 1234y; // expected-error-re {{no matching literal operator for call to 'operator""y' {{.*}}}}

  return 0;
}
