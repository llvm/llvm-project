//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

#include <random>

void test()
{
  {
    std::binomial_distribution<bool> baddist; //expected-error@*:* {{IntType must be a supported integer type}}
    std::binomial_distribution<int> okdist;
    (void)baddist;
    (void)okdist;
  }
  {
    std::discrete_distribution<bool> baddist; //expected-error@*:* {{IntType must be a supported integer type}}
    std::discrete_distribution<int> okdist;
    (void)baddist;
    (void)okdist;
  }
  {
    std::geometric_distribution<bool> baddist; //expected-error@*:* {{IntType must be a supported integer type}}
    std::geometric_distribution<int> okdist;
    (void)baddist;
    (void)okdist;
  }
  {
    std::negative_binomial_distribution<bool> baddist; //expected-error@*:* {{IntType must be a supported integer type}}
    std::negative_binomial_distribution<int> okdist;
    (void)baddist;
    (void)okdist;
  }
  {
    std::poisson_distribution<bool> baddist; //expected-error@*:* {{IntType must be a supported integer type}}
    std::poisson_distribution<int> okdist;
    (void)baddist;
    (void)okdist;
  }
  {
    std::uniform_int_distribution<bool> baddist; //expected-error@*:* {{IntType must be a supported integer type}}
    std::uniform_int_distribution<int> okdist;
    (void)baddist;
    (void)okdist;
  }
}
