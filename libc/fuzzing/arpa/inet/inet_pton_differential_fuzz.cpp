//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Differential fuzz test for llvm-libc inet_pton implementation.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/scope.h"
#include "src/arpa/inet/inet_pton.h"
#include <arpa/inet.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 1)
    return 0;

  uint8_t af_selector = data[0];
  int af = (af_selector & 0x80) ? AF_INET : AF_INET6;

  // Create a null-terminated copy of the string to parse.
  size_t str_size = size - 1;
  char *str = new char[str_size + 1];
  LIBC_NAMESPACE::cpp::scope_exit delete_str([&] { delete[] str; });
  memcpy(str, data + 1, str_size);
  str[str_size] = '\0';

  // Setup destination buffers.
  constexpr size_t BUFFER_SIZE = 64;
  char ref_dst[BUFFER_SIZE];
  char impl_dst[BUFFER_SIZE];

  constexpr uint8_t SENTINEL = 0x5a;
  memset(ref_dst, SENTINEL, BUFFER_SIZE);
  memset(impl_dst, SENTINEL, BUFFER_SIZE);

  // Call reference implementation.
  int ref_res = ::inet_pton(af, str, ref_dst);

  // Call our implementation.
  int impl_res = LIBC_NAMESPACE::inet_pton(af, str, impl_dst);

  size_t addr_size =
      (af == AF_INET) ? sizeof(struct in_addr) : sizeof(struct in6_addr);

  auto print_details = [&]() {
    fprintf(stderr,
            "Details:\n"
            "  af: %d (%s)\n"
            "  str: \"%s\"\n"
            "  ref_res: %d\n"
            "  impl_res: %d\n",
            af, (af == AF_INET) ? "AF_INET" : "AF_INET6", str, ref_res,
            impl_res);
    if (ref_res == 1) {
      fprintf(stderr, "  ref_addr: ");
      for (size_t i = 0; i < addr_size; ++i)
        fprintf(stderr, "%02x", static_cast<uint8_t>(ref_dst[i]));
      fprintf(stderr, "\n  impl_addr: ");
      for (size_t i = 0; i < addr_size; ++i)
        fprintf(stderr, "%02x", static_cast<uint8_t>(impl_dst[i]));
      fprintf(stderr, "\n");
    }
  };

  // Compare results.
  if (ref_res != impl_res) {
    fprintf(stderr, "Success/failure mismatch!\n");
    print_details();
    __builtin_trap();
  }

  if (ref_res == 1) {
    // Both succeeded. Check that parsed addresses match.
    if (memcmp(ref_dst, impl_dst, addr_size) != 0) {
      fprintf(stderr, "Parsed address mismatch!\n");
      print_details();
      __builtin_trap();
    }
  }

  // Check for out-of-bounds writes.
  for (size_t i = addr_size; i < BUFFER_SIZE; ++i) {
    if (static_cast<uint8_t>(impl_dst[i]) != SENTINEL) {
      fprintf(stderr,
              "Out-of-bounds write detected at index %zu (expected 0x%02x, got "
              "0x%02x)!\n",
              i, SENTINEL, static_cast<uint8_t>(impl_dst[i]));
      print_details();
      __builtin_trap();
    }
  }

  return 0;
}
