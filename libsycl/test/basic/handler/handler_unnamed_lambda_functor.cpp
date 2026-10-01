//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-device-only -std=c++17 -fsyntax-only %s

#include <sycl/sycl.hpp>

struct SingleTaskKernel {
  void operator()() const {}
};

int main() {
  sycl::queue q;

  q.single_task<class QueueSingleTaskNamed>(SingleTaskKernel{});

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class HandlerSingleTaskNamed>(SingleTaskKernel{});
  });

  return 0;
}
