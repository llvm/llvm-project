//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that we can provide function pointers as input to
// std::find_if. The OpenMP declare target directive with the `indirect` clause
// makes an implicit mapping of the host and device function pointers.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

bool is_odd(int& i) { return (i % 2) == 1; }
#pragma omp declare target indirect to(is_odd)

int main(int, char**) {
  const int test_size = 10000;
  std::vector<int> v(test_size, 2.0);
  v[123] = 3;

  // Providing for_each a function pointer
  auto idx = std::find_if(std::execution::par_unseq, v.begin(), v.end(), is_odd);

  assert(idx - v.begin() == 123 && "std::find_if(std::execution::par_unseq,...) does not accept function pointers");
  return 0;
}
