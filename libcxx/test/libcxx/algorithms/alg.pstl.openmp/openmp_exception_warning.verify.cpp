//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test validates that the user is prompted with a warning if they use
// exception handling inside a function called in an offloaded parallel
// algorithm.

// This test must be compiled with --offload-device-only to avoid the verify
// expects warnings for the host fall-back code.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -Wopenmp-target-exception -fexceptions --offload-arch=native --offload-device-only

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>
#include <execution>
#include <vector>

bool is_odd(int& i) {
  try { // expected-warning {{does not support exception handling; 'catch' block is ignored}}
    if (i % 2 == 0) {
      return true;
    } else {
      throw false; // expected-warning {{does not support exception handling; 'throw' is assumed to be never reached}}
    }
  } catch (bool b) {
    return b;
  }
}
#pragma omp declare target indirect to(is_odd)

int main(int, char**) {
  const int test_size = 10000;
  std::vector<int> v(test_size, 2.0);

  // Providing find_if a function pointer
  std::find_if(std::execution::par_unseq, v.begin(), v.end(), is_odd);
  return 0;
}
