//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

#include <random>

void test() {
  {
    std::uniform_real_distribution<int>
        baddist; //expected-error@*:* {{RealType must be a supported floating-point type}}
    std::uniform_real_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }
  {
    std::exponential_distribution<int>
        baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::exponential_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::gamma_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::gamma_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::weibull_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::weibull_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::extreme_value_distribution<int>
        baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::extreme_value_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::normal_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::normal_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::lognormal_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::lognormal_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::chi_squared_distribution<int>
        baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::chi_squared_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::cauchy_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::cauchy_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::fisher_f_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::fisher_f_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::student_t_distribution<int> baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::student_t_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::piecewise_constant_distribution<int>
        baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::piecewise_constant_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }

  {
    std::piecewise_linear_distribution<int>
        baddist; // expected-error@*:* {{RealType must be a supported floating-point type}}
    std::piecewise_linear_distribution<double> okdist;
    (void)baddist;
    (void)okdist;
  }
}
