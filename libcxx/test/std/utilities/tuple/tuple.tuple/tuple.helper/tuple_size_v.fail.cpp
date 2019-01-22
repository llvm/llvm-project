//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <tuple>

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

// Expect failures with a reference type, pointer type, and a non-tuple type.

#include <tuple>

int main()
{
    (void)std::tuple_size_v<std::tuple<> &>; // expected-note {{requested here}}
    (void)std::tuple_size_v<int>; // expected-note {{requested here}}
    (void)std::tuple_size_v<std::tuple<>*>; // expected-note {{requested here}}
    // expected-error@tuple:* 3 {{implicit instantiation of undefined template}}
}
