//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-5, clang-6
// UNSUPPORTED: apple-clang-9, apple-clang-10

// <chrono>
// class day;

// constexpr day operator""d(unsigned long long d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using day = std::chrono::day;
    day d1 = 4d; // expected-error-re {{no matching literal operator for call to 'operator""d' {{.*}}}}

  return 0;
}
