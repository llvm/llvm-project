//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but syd::find_if(std::execution::par_unseq,...) is not executed on
// the device.

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
  std::vector<double> v(test_size, 1);

  auto idx = std::find_if(std::execution::par_unseq, v.begin(), v.end(), [](double&) -> bool {
    // Returns true if executed on the host
    return omp_is_initial_device();
  });
  assert(idx == v.end() &&
         "omp_is_initial_device() returned true in the target region. std::find_if was not offloaded.");
  return 0;
}
