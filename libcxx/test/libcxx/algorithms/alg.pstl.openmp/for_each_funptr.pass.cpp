//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that we can provide function pointers as input to
// std::for_each. The OpenMP declare target directive with the `indirect` clause
// makes an implicit mapping of the host and device function pointers.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

void cube(double& d) { d *= d * d; }
#pragma omp declare target indirect to(cube)

int main(int, char**) {
  const int test_size = 10000;
  std::vector<double> v(test_size, 2.0);

  // Providing for_each a function pointer
  std::for_each(std::execution::par_unseq, v.begin(), v.end(), cube);

  for (int vi : v)
    assert(vi == 8 && "std::for_each(std::execution::par_unseq,...) does not accept function pointers");
  return 0;
}
