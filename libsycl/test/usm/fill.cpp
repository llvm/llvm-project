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

#include "Inputs/fill_memset_common.hpp"

struct Foo {
  unsigned char Val[3];
  bool operator!=(const Foo &Rhs) const {
    for (std::size_t I = 0; I < 3; ++I)
      if (Val[I] != Rhs.Val[I])
        return true;
    return false;
  }
};

int main() {
  sycl::queue Q;
  runTests<int>(
      Q, [&](void *Ptr, int Pattern) { Q.fill(Ptr, Pattern, ElementCount); });
  // Liboffload handles patterns with a size that's not a power of two
  // differently, check that case separately.
  Foo Val({'a', 'b', 'c'});
  runTests<Foo>(
      Q, [&](void *Ptr, Foo Pattern) { Q.fill(Ptr, Pattern, ElementCount); },
      Val);
}
