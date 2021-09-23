//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// tuple_element<I, array<T, N> >::type

// Prevent -Warray-bounds from issuing a diagnostic when testing with clang verify.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Warray-bounds"
#endif

#include <array>
#include <cassert>

int main(int, char**)
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        std::tuple_element<3, C> foo; // expected-note {{requested here}}
        // expected-error-re@array:* {{static_assert failed{{( due to requirement '3U[L]{0,2} < 3U[L]{0,2}')?}} "Index out of bounds in std::tuple_element<> (std::array)"}}
    }

  return 0;
}
