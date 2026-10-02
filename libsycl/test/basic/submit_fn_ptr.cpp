//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl  %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

class Test;

int main() {
  sycl::queue q;
  int *p = sycl::malloc_shared<int>(1, q);
  *p = 0;
  q.single_task<Test>([=]() { *p = 42; });
  q.wait();

  bool Failed = *p != 42;

  sycl::free(p, q);
  return Failed;
}
