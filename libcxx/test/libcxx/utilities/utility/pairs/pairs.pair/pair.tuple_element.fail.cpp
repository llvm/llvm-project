//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <utility>

int main()
{
    {
    typedef std::pair<int, double> P;
    std::tuple_element<2, P>::type foo; // expected-note {{requested here}}
        // expected-error-re@utility:* {{static_assert failed{{( due to requirement '2U[L]{0,2} < 2')?}} "Index out of bounds in std::tuple_element<std::pair<T1, T2>>"}}
    }
}
