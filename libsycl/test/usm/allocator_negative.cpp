//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -fsyntax-only -Xclang -verify \
// RUN: -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

void testDeviceAllocKind(sycl::queue &Q) {
  sycl::usm_allocator<int, sycl::usm::alloc::device> Alloc(Q);
  // expected-error@*:* {{usm_allocator does not support AllocKind == usm::alloc::device}}
  (void)Alloc;
}
