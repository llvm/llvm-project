//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution
// {
//     class param_type;

#include <random>
#include <limits>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::bernoulli_distribution D;
        typedef D::param_type param_type;
        param_type p;
        assert(p.p() == 0.5);
    }
    {
        typedef std::bernoulli_distribution D;
        typedef D::param_type param_type;
        param_type p(0.25);
        assert(p.p() == 0.25);
    }

  return 0;
}
