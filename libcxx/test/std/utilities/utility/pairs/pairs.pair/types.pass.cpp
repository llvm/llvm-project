//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2>
// struct pair
// {
//     typedef T1 first_type;
//     typedef T2 second_type;

#include <utility>
#include <type_traits>

int main(int, char**)
{
    typedef std::pair<float, short*> P;
    static_assert((std::is_same<P::first_type, float>::value), "");
    static_assert((std::is_same<P::second_type, short*>::value), "");

  return 0;
}
