//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept

// An overload exists for each of char and all arithmetic types except bool.

#include <numeric>

#include "test_macros.h"

int main(int, char**)
{
    (void) std::midpoint(false, true); // expected-error {{no matching function for call to 'midpoint'}}

//  A couple of odd pointer types that should fail
    (void) std::midpoint(nullptr, nullptr);     // expected-error {{no matching function for call to 'midpoint'}}
    (void) std::midpoint((void *)0, (void *)0); // expected-error@numeric:* {{arithmetic on pointers to void}}
    
    return 0;
}
