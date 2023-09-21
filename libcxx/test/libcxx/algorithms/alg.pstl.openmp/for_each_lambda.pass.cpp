//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that we can provide a lambda as input to std::for_each in
// different ways.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

template <class Function, class Tp>
void test_lambda(Function fun, Tp initial_value, Tp final_value) {
  const int test_size = 10000;
  std::vector<double> v(test_size, initial_value);

  // Providing for_each a function pointer
  std::for_each(std::execution::par_unseq, v.begin(), v.end(), fun);

  for (int vi : v)
    assert(vi == final_value && "std::for_each(std::execution::par_unseq,...) does not accept lambdas");
}

int main(int, char**) {
  // Capturing by reference
  auto cube_ref = [&](double& a) { a *= a * a; };
  test_lambda(cube_ref, 2.0, 8.0);

  // Capturing by value
  auto cube_val = [=](double& a) { a *= a * a; };
  test_lambda(cube_val, 2.0, 8.0);

  // Capturing by reference when using additional input
  double c1       = 1.0;
  auto cube_ref_2 = [&](double& a) { a = a * a * a + c1; };
#pragma omp target data map(to : c1)
  test_lambda(cube_ref_2, 2.0, 9.0);
  return 0;
}
