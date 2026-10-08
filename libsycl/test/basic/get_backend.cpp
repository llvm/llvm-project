//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

class Kernel1;

bool check(backend Backend) {
  switch (Backend) {
  case backend::opencl:
  case backend::level_zero:
  case backend::cuda:
  case backend::hip:
    return true;
  default:
    return false;
  }
}

int main() {
  for (const auto &Plt : platform::get_platforms()) {
    if (!check(Plt.get_backend()))
      return 1;

    auto Dev = Plt.get_devices()[0];
    if (Dev.get_backend() != Plt.get_backend())
      return 1;

    queue Q(Dev);
    if (Q.get_backend() != Plt.get_backend())
      return 1;

    event E = Q.single_task<Kernel1>([]() {});
    if (E.get_backend() != Plt.get_backend())
      return 1;
  }
  return 0;
}
