//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but std::for_each(std::execution::par_unseq,...) is not executed on
// the device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: --offload-arch=native

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

int main(int, char**) {
  // Initializing test array
  const int test_size = 10000;
  std::vector<int> v(test_size, 2);

  // By making an extra map, we can control when the data is mapped to and from
  // the device, because the map inside std::fill will then only increment and
  // decrement reference counters and not move data.
  int* data = v.data();
#pragma omp target enter data map(to : data[0 : v.size()])
  std::fill(std::execution::par_unseq, v.begin(), v.end(), -2);

  // At this point v should only contain the value 2
  for (int vi : v)
    assert(vi == 2 &&
           "std::fill transferred data from device to the host but should only have decreased the reference counter.");

// After moving the result back to the host it should now be -2
#pragma omp target update from(data[0 : v.size()])
  for (int vi : v)
    assert(vi == -2 && "std::fill did not update the result on the device.");

#pragma omp target exit data map(delete : data[0 : v.size()])

  return 0;
}
