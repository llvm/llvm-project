//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Differential fuzz test for llvm-libc inet_aton implementation.
///
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_aton.h"
#include <arpa/inet.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size == 0 || data[size - 1] != '\0')
    return -1;

  struct in_addr ref_addr = {};
  struct in_addr impl_addr = {};
  int ref = ::inet_aton(reinterpret_cast<const char *>(data), &ref_addr);
  int impl = LIBC_NAMESPACE::inet_aton(reinterpret_cast<const char *>(data),
                                       &impl_addr);

  if (ref != impl) {
    fprintf(stderr,
            "Different result (reference: %d, implementation: %d) for \"%s\"\n",
            ref, impl, data);
    __builtin_trap();
  }

  if (ref == 1 && ref_addr.s_addr != impl_addr.s_addr) {
    fprintf(
        stderr,
        "Different address (reference: %x, implementation: %x) for \"%s\"\n",
        ref_addr.s_addr, impl_addr.s_addr, data);
    __builtin_trap();
  }

  return 0;
}
