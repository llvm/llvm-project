//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but std::transform(std::execution::par_unseq,...) is not executed
// on the device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

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

  // Initializing test arrays
  const int test_size = 10000;
  std::vector<int> host(test_size);
  std::vector<int> device(test_size);
  // Should execute on host
  std::transform(std::execution::unseq, host.begin(), host.end(), host.begin(), [](int& h) {
    // Returns true if executed on the host
    h = omp_is_initial_device();
    return h;
  });

  // Asserting the std::transform(std::execution::unseq,...) executed on the host
  for (int hi : host)
    assert(hi && "omp_is_initial_device() returned false. std::transform was offloaded but shouldn't be.");

  // Should execute on device
  std::transform(
      std::execution::par_unseq, device.begin(), device.end(), host.begin(), device.begin(), [](int& d, int& h) {
        // Should return fals
        d = omp_is_initial_device();
        return h == d;
      });

  // Asserting the std::transform(std::execution::par_unseq,...) executed on the device
  for (int di : device)
    assert(!di && "omp_is_initial_device() returned true in the target region. std::transform was not offloaded.");
  return 0;
}
