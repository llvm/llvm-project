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

#include "src/__support/CPP/scope.h"
#include "src/arpa/inet/inet_aton.h"
#include "src/string/memcpy.h"
#include <arpa/inet.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Create a null-terminated copy of the data
  char *str = new char[size + 1];
  LIBC_NAMESPACE::cpp::scope_exit delete_str([&] { delete[] str; });
  LIBC_NAMESPACE::memcpy(str, data, size);
  str[size] = '\0';

  struct in_addr ref_addr = {};
  struct in_addr impl_addr = {};
  int ref = ::inet_aton(str, &ref_addr);
  int impl = LIBC_NAMESPACE::inet_aton(str, &impl_addr);

  if (ref != impl) {
    fprintf(stderr,
            "Different result (reference: %d, implementation: %d) for \"%s\"\n",
            ref, impl, str);
    __builtin_trap();
  }

  if (ref == 1 && ref_addr.s_addr != impl_addr.s_addr) {
    fprintf(
        stderr,
        "Different address (reference: %x, implementation: %x) for \"%s\"\n",
        ref_addr.s_addr, impl_addr.s_addr, str);
    __builtin_trap();
  }

  return 0;
}
