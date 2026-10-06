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
    auto CtxDevs = P.khr_get_default_context().get_devices();
    auto RootDevs = P.get_devices();

    for (const auto &Dev : RootDevs)
      if (std::find(CtxDevs.begin(), CtxDevs.end(), Dev) == CtxDevs.end())
        return 1;
  }

  return 0;
}
