//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but std::transform_reduce(std::execution::par_unseq,...) is not
// executed on the device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <execution>
#include <functional>
#include <vector>
#include <omp.h>

int main(int, char**) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  // Initializing test array
  const int test_size = 10000;
  std::vector<int> v(test_size, 1);
  std::vector<int> w(test_size, 1);

  int result = std::transform_reduce(
      std::execution::par_unseq, v.begin(), v.end(), w.begin(), (int)0, std::plus{}, [](int& n, int& m) {
        return n + m + omp_is_initial_device(); // Gives 2 if executed on device, 3 if executed on host
      });
  assert(result == 2 * test_size &&
         "omp_is_initial_device() returned true in the target region. std::transform_reduce was not offloaded.");
  return 0;
}
