//===-- harness.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/tests/harness.h"

namespace gwp_asan {
namespace test {
bool OnlyOnce() {
  static int x = 0;
  return !x++;
}
} // namespace test
} // namespace gwp_asan

// Optnone to ensure that the calls to these functions are not optimized away,
// as we're looking for them in the backtraces.
__attribute__((optnone)) char *
AllocateMemory(gwp_asan::GuardedPoolAllocator &GPA) {
  return static_cast<char *>(GPA.allocate(1));
}
__attribute__((optnone)) void
DeallocateMemory(gwp_asan::GuardedPoolAllocator &GPA, void *Ptr) {
  GPA.deallocate(Ptr);
}
__attribute__((optnone)) void
DeallocateMemory2(gwp_asan::GuardedPoolAllocator &GPA, void *Ptr) {
  GPA.deallocate(Ptr);
}
__attribute__((optnone)) void TouchMemory(void *Ptr) {
  *(reinterpret_cast<volatile char *>(Ptr)) = 7;
}
