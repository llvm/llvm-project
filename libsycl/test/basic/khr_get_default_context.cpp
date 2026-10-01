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

// Test checks that the default context contains all of the root devices that
// are associated with this platform.

#include <sycl/sycl.hpp>

#include <algorithm>

using namespace sycl;

int main() {
  for (const sycl::platform &P : sycl::platform::get_platforms()) {
    auto ctx_devs = P.khr_get_default_context().get_devices();
    auto root_devs = P.get_devices();

    for (const auto &dev : root_devs)
      if (std::find(ctx_devs.begin(), ctx_devs.end(), dev) == ctx_devs.end())
        return 1;
  }

  return 0;
}
