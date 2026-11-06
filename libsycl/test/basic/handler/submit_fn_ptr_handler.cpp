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

#include <cassert>

int *Ptr = nullptr;

void foo(sycl::handler &CGH) {
  auto *Copy = Ptr;
  CGH.single_task([=]() { *Copy = 42; });
}

int main() {
  sycl::queue Q;
  Ptr = sycl::malloc_shared<int>(1, Q);
  *Ptr = 0;
  Q.submit(foo).wait();
  assert(*Ptr == 42);
  sycl::free(Ptr, Q);
  return 0;
}
