//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// Period shall be a specialization of ratio, diagnostic required.

#include <chrono>

template <int N, int D = 1>
class Ratio
{
public:
    static const int num = N;
    static const int den = D;
};

int main(int, char**)
{
    typedef std::chrono::duration<int, Ratio<1> > D;
    D d;

  return 0;
}
