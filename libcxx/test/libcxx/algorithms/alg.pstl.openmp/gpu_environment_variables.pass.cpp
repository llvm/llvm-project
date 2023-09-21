//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that the libc++ test configuration forwards the AMD and
// NVIDIA environment variables specifying the visible devices. Intially when
// developing the OpenMP offloading tests, this was not the case, and this test
// will reveal if the configuration is wrong another time.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// REQUIRES: libcpp-pstl-backend-openmp

#include <string>
#include <cassert>
#include <omp.h>
#include <iostream>

std::string get_env_var(std::string const& env_var_name, int& flag) {
  char* val;
  val                = getenv(env_var_name.c_str());
  std::string retval = "";
  flag               = (val != NULL);
  return (val != NULL) ? val : "";
}

int main(int, char**) {
  // Stores whether the environment variable was found
  int status = 0;

  // Checking for AMD's enviroment variable for specifying visible devices
  std::string rocr_visible_devices = get_env_var("ROCR_VISIBLE_DEVICES", status);
  if (status)
    assert(
        (rocr_visible_devices.empty() || (omp_get_num_devices() > 0)) &&
        "ROCR_VISIBLE_DEVICES was set but no devices were detected by OpenMP. The libc++ test suite is misconfigured.");

  // Checking for NVIDIA's enviroment variable for specifying visible devices
  std::string cuda_visible_devices = get_env_var("CUDA_VISIBLE_DEVICES", status);
  if (status)
    assert(
        (cuda_visible_devices.empty() || (omp_get_num_devices() > 0)) &&
        "CUDA_VISIBLE_DEVICES was set but no devices were detected by OpenMP. The libc++ test suite is misconfigured.");
  return 0;
}
