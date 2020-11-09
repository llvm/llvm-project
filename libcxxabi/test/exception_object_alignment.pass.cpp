//===---------------- exception_object_alignment.pass.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// Check that the pointer __cxa_allocate_exception returns is aligned to the
// default alignment for the target architecture.

#include <cassert>
#include <cstdint>
#include <cxxabi.h>
#include <type_traits>
#include <__cxxabi_config.h>

struct S {
  int a[4];
} __attribute__((aligned));

int main(int, char**) {
#if !defined(_LIBCXXABI_ARM_EHABI)
  void *p = __cxxabiv1::__cxa_allocate_exception(16);
  auto i = reinterpret_cast<uintptr_t>(p);
  auto a = std::alignment_of<S>::value;
  assert(i % a == 0);
  __cxxabiv1::__cxa_free_exception(p);
#endif
  return 0;
}
