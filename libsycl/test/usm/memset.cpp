//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: any-device
// https://github.com/llvm/llvm-project/issues/225092
// UNSUPPORTED: true
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include "Inputs/fill_memset_common.hpp"

#include <climits>

int main() {
  sycl::queue Q;
  runTests<unsigned char>(
      Q, [&](void *Ptr, int Pattern) { Q.memset(Ptr, Pattern, ElementCount); });
  // Check that the pattern is truncated to an unsigned char.
  runTests<unsigned char>(
      Q, [&](void *Ptr, int Pattern) { Q.memset(Ptr, Pattern, ElementCount); },
      CHAR_MAX + 42);
}
