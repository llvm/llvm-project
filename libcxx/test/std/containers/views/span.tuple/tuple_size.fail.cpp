//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// tuple_size<span<T, N> >::value

#include <span>

int main(int, char**)
{
    (void) std::tuple_size<std::span<double>>::value; // expected-error-re {{implicit instantiation of undefined template 'std:{{.*}}:tuple_size<std:{{.*}}:span<double, {{.*}}}}
    (void) std::tuple_size<std::span<int>>::value;    // expected-error-re {{implicit instantiation of undefined template 'std:{{.*}}:tuple_size<std:{{.*}}:span<int, {{.*}}}}
    return 0;
}
