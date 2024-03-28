//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that you can overwrite the input in
// std::for_each(std::execution::par_unseq,...). If the result was not copied
// back from the device to the host, this test would fail.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <vector>

template <class _Tp, class _Predicate, class _Up>
void overwrite(_Tp& data, _Predicate pred, const _Up& value) {
  // This function assumes that pred will never be the identity transformation

  // Updating the array with a lambda
  std::for_each(std::execution::par_unseq, data.begin(), data.end(), pred);

  // Asserting that no elements have the intial value
  for (int di : data)
    assert(
        di != value &&
        "The GPU implementation of std::for_each does not allow users to mutate the input as the C++ standard does.");
}

int main(int, char**) {
  const double value  = 2.0;
  const int test_size = 10000;
  // Testing with vector of doubles
  {
    std::vector<double> v(test_size, value);
    overwrite(v, [&](double& n) { n *= n; }, value);
  }
  // Testing with vector of integers
  {
    std::vector<int> v(test_size, (int)value);
    overwrite(v, [&](int& n) { n *= n; }, (int)value);
  }
  // Testing with array of doubles
  {
    std::array<double, test_size> a;
    a.fill(value);
    overwrite(a, [&](double& n) { n *= n; }, value);
  }
  // Testing with array of integers
  {
    std::array<int, test_size> a;
    a.fill((int)value);
    overwrite(a, [&](int& n) { n *= n; }, (int)value);
  }
  return 0;
}
