//===-- Implementation of apply_irelative_relocs test ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(ifunc)

// Two trivial implementations that return different constants.
static int impl_a() { return 42; }
static int impl_b() { return 84; }

// The resolver function. On x86_64 this takes no arguments; on AArch64 it
// receives (hwcap, hwcap2). We use extern "C" to ensure C linkage for the
// ifunc attribute.
extern "C" {
// Declare the resolver to return a generic function pointer.
// For testing, unconditionally select impl_a.
void *my_ifunc_resolver() {
  (void)impl_b; // Suppress unused warning.
  return reinterpret_cast<void *>(impl_a);
}
}

// Declare the ifunc. The compiler and linker will generate an IRELATIVE
// relocation that calls my_ifunc_resolver at startup.
extern "C" int my_ifunc() __attribute__((ifunc("my_ifunc_resolver")));

TEST_MAIN() {
  // If IRELATIVE processing works correctly, my_ifunc() should call impl_a
  // and return 42.
  ASSERT_EQ(my_ifunc(), 42);
  return 0;
}

#else

TEST_MAIN() { return 0; }

#endif
