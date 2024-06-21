//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but for_each(std::execution::par_unseq,...) is not executed on the
// device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

int main(int, char**) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  // Initializing test array
  const int test_size = 10000;
  std::vector<int> v(test_size);
  std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](int& n) {
    // Returns true if executed on the host
    n = omp_is_initial_device();
  });

  for (int vi : v)
    assert(vi == 0 && "omp_is_initial_device() returned true in the target region. std::for_each was not offloaded.");
  return 0;
}
