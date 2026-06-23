//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// Check that functions are marked [[nodiscard]]

#include <random>

#include "test_macros.h"

void test() {
  std::mt19937_64 gen;

  {
    std::bernoulli_distribution::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.p();

    std::bernoulli_distribution d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.p();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::binomial_distribution<int>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.t();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.p();

    std::binomial_distribution<int> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.p();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::cauchy_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.b();

    std::cauchy_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.b();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::chi_squared_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.n();

    std::chi_squared_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.n();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::discard_block_engine<std::mt19937_64, 10, 5> e;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.max();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.base();
  }
  {
    std::discrete_distribution<int>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.probabilities();

    std::discrete_distribution<int> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.probabilities();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::exponential_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.lambda();

    std::exponential_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.lambda();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::extreme_value_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.b();

    std::extreme_value_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.b();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::fisher_f_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.m();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.n();

    std::fisher_f_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.m();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.n();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::gamma_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.alpha();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.beta();

    std::gamma_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.alpha();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.beta();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::generate_canonical<double, 10>(gen);
  }
  {
    std::geometric_distribution<int>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.p();

    std::geometric_distribution<int> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.p();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::independent_bits_engine<std::mt19937_64, 10, unsigned int> e;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.max();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.base();
  }
  {
    std::linear_congruential_engine<unsigned int, 48271, 0, 2147483647> e(94);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.max();
  }
  {
    std::lognormal_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.m();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.s();

    std::lognormal_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.m();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.s();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::mt19937_64 e;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.max();
  }
  {
    std::negative_binomial_distribution<int>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.k();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.p();

    std::negative_binomial_distribution<int> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.k();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.p();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::normal_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.mean();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.stddev();

    std::normal_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.mean();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.stddev();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::piecewise_constant_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.intervals();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.densities();

    std::piecewise_constant_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.intervals();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.densities();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::piecewise_linear_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.intervals();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.densities();

    std::piecewise_linear_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.intervals();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.densities();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::poisson_distribution<int>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.mean();

    std::poisson_distribution<int> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.mean();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
#if !defined(TEST_HAS_NO_RANDOM_DEVICE)
  {
    std::random_device d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.entropy();
  }
#endif
  {
    std::seed_seq s;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    s.size();
  }
  {
    std::shuffle_order_engine<std::mt19937_64, 10> e;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.max();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.base();
  }
  {
    std::student_t_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.n();

    std::student_t_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.n();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::subtract_with_carry_engine<unsigned int, 24, 10, 24> e;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    e.max();
  }
  {
    std::uniform_int_distribution<int>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.b();

    std::uniform_int_distribution<int> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.b();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::uniform_real_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.b();

    std::uniform_real_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.b();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
  {
    std::weibull_distribution<double>::param_type p;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.b();

    std::weibull_distribution<double> d;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d(gen, p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.a();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.b();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.param();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.min();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    d.max();
  }
}
