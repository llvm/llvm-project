//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that std::transform_reduce(std::execution::par_unseq,...)
// can be offloaded for a number of supported binary operations. The following
// binary operations should be supported for the reducer:
// - std::plus
// - std::minus
// - std::multiplies
// - std::logical_and
// - std::logical_or
// - std::bit_and
// - std::bit_or
// - std::bit_xor

// UNSUPPORTED: c++03, c++11, c++14, gcc

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <execution>
#include <functional>
#include <vector>
#include <omp.h>
#include <iostream>

int main(int, char**) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  // Initializing test array
  const int test_size = 10000;

  //===--------------------------------------------------------------------===//
  // Arithmetic binary operators
  //===--------------------------------------------------------------------===//

  // Addition with doubles
  {
    std::vector<double> v(test_size, 1.0);
    std::vector<double> w(test_size, 2.0);
    double result = std::transform_reduce(
        std::execution::par_unseq, v.begin(), v.end(), w.begin(), 5.0, std::plus{}, [](double& a, double& b) {
          return 0.5 * (b - a) * ((double)!omp_is_initial_device());
        });
    assert((std::abs(result - 0.5 * ((double)test_size) - 5.0) < 1e-8) &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the binary "
           "operation std::plus.");
  }

  // Subtraction of floats
  {
    std::vector<float> v(test_size, 1.0f);
    std::vector<float> w(test_size, 1.5f);
    float result = std::transform_reduce(
        std::execution::par_unseq,
        v.begin(),
        v.end(),
        w.begin(),
        1.25 * ((float)test_size),
        std::minus{},
        [](float& a, float& b) { return 0.5 * (a + b) * ((float)!omp_is_initial_device()); });
    assert((std::abs(result) < 1e-8f) &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the "
           "intended effect for the binary operation std::minus.");
  }

  // Multiplication of doubles
  {
    std::vector<double> v(test_size, 1.0);
    std::vector<double> w(test_size, 0.0001);
    double result = std::transform_reduce(
        std::execution::par_unseq, v.begin(), v.end(), w.begin(), -1.0, std::multiplies{}, [](double& a, double& b) {
          return (a + b) * ((double)!omp_is_initial_device());
        });
    assert((std::abs(result + pow(1.0001, test_size)) < 1e-8) &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the binary "
           "operation std::multiplies.");
  }

  //===--------------------------------------------------------------------===//
  // Logical binary operators
  //===--------------------------------------------------------------------===//

  // Logical and
  {
    std::vector<int> v(test_size, 1);
    // The result should be true with an initial value of 1
    int result =
        std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 1, std::logical_and{}, [](int& a) {
          return a && !omp_is_initial_device();
        });
    assert(result &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the binary "
           "operation std::logical_and.");

    // And false by an initial value of 0
    result = std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 0, std::logical_and{}, [](int& a) {
      return a && !omp_is_initial_device();
    });
    assert(!result &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the binary "
           "operation std::logical_and.");
  }

  // Logical or
  {
    std::vector<int> v(test_size, 0);
    // The result should be true with an initial value of 1
    int result = std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 1, std::logical_or{}, [](int& a) {
      return a && !omp_is_initial_device();
    });
    assert(result &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the binary "
           "operation std::logical_or.");

    // And false by an initial value of 0
    result = std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 0, std::logical_or{}, [](int& a) {
      return a && !omp_is_initial_device();
    });
    assert(!result && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the "
                      "binary operation std::logical_or.");
  }

  //===--------------------------------------------------------------------===//
  // Bitwise binary operators
  //===--------------------------------------------------------------------===//

  // Bitwise and
  {
    std::vector<unsigned int> v(test_size, 3);
    std::vector<unsigned int> w(test_size, 2);
    // For odd numbers the result should be true
    int result =
        std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 0x1, std::bit_and{}, [](unsigned int& a) {
          return a + omp_is_initial_device();
        });
    assert(result && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the "
                     "binary operation std::bit_and.");

    // For even numbers the result should be false
    result =
        std::transform_reduce(std::execution::par_unseq, w.begin(), w.end(), 0x1, std::bit_and{}, [](unsigned int& a) {
          return a + omp_is_initial_device();
        });
    assert(!result && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the "
                      "binary operation std::bit_and.");
  }

  // Bitwise or
  {
    std::vector<unsigned int> v(test_size, 0);
    int result = std::transform_reduce(
        std::execution::par_unseq, v.begin(), v.end(), 0, std::bit_or{}, [](unsigned int& a) -> unsigned int {
          return a || omp_is_initial_device();
        });
    assert(!result && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the "
                      "binary operation std::bit_or.");

    // After adding a one, the result should be true
    v[v.size() / 2] = 1;
    result          = std::transform_reduce(
        std::execution::par_unseq, v.begin(), v.end(), 0, std::bit_or{}, [](unsigned int& a) -> unsigned int {
          return a && !omp_is_initial_device();
        });
    assert(result && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the "
                     "binary operation std::bit_or.");
  }

  // Bitwise xor
  {
    std::vector<unsigned int> v(test_size, 0xef);
    int result =
        std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 0, std::bit_xor{}, [](unsigned int& a) {
          return a << omp_is_initial_device();
        });
    assert(result == 0 && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for "
                          "the binary operation std::bit_or.");

    // After adding a one, the result should be true
    v[v.size() / 2] = 0xea;
    result =
        std::transform_reduce(std::execution::par_unseq, v.begin(), v.end(), 0, std::bit_xor{}, [](unsigned int& a) {
          return a << omp_is_initial_device();
        });
    assert(result == 5 && "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for "
                          "the binary operation std::bit_or.");
  }

  return 0;
}
