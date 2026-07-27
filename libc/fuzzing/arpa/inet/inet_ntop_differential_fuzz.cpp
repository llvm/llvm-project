//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Differential fuzz test for llvm-libc inet_ntop implementation.
///
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_ntop.h"
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
  socklen_t dst_size = data[0] % 64;

  // Extract address bytes. We allocate 16 bytes for IPv6, IPv4 will only use 4.
  uint8_t address_bytes[16] = {0};
  size_t payload_size = size - 1;
  size_t copy_size = payload_size > 16 ? 16 : payload_size;
  memcpy(address_bytes, data + 1, copy_size);

  // Setup buffers for dst
  constexpr size_t BUFFER_SIZE = 128;
  char ref_dst[BUFFER_SIZE];
  char impl_dst[BUFFER_SIZE];

  constexpr uint8_t SENTINEL = 0x5a;
  memset(ref_dst, SENTINEL, BUFFER_SIZE);
  memset(impl_dst, SENTINEL, BUFFER_SIZE);

  // Call reference implementation
  const char *ref_res = ::inet_ntop(af, address_bytes, ref_dst, dst_size);

  // Call our implementation
  const char *impl_res =
      LIBC_NAMESPACE::inet_ntop(af, address_bytes, impl_dst, dst_size);

  auto print_details = [&]() {
    fprintf(stderr,
            "Details:\n"
            "  af: %d\n"
            "  dst_size: %lu\n"
            "  address_bytes: %02x%02x %02x%02x %02x%02x %02x%02x %02x%02x "
            "%02x%02x %02x%02x %02x%02x\n"
            "  ref_res: %s\n"
            "  impl_res: %s\n",
            af, static_cast<unsigned long>(dst_size), address_bytes[0],
            address_bytes[1], address_bytes[2], address_bytes[3],
            address_bytes[4], address_bytes[5], address_bytes[6],
            address_bytes[7], address_bytes[8], address_bytes[9],
            address_bytes[10], address_bytes[11], address_bytes[12],
            address_bytes[13], address_bytes[14], address_bytes[15],
            ref_res ? ref_res : "nullptr", impl_res ? impl_res : "nullptr");
  };

  // Compare results
  if ((ref_res == nullptr) != (impl_res == nullptr)) {
    fprintf(stderr, "Success/failure mismatch!\n");
    print_details();
    __builtin_trap();
  }

  if (ref_res != nullptr) {
    // Both succeeded.
    // Check that returned pointers are correct
    if (ref_res != ref_dst || impl_res != impl_dst) {
      fprintf(stderr, "Returned pointer does not match destination buffer!\n");
      print_details();
      __builtin_trap();
    }
    // Check that strings match
    if (strcmp(ref_res, impl_res) != 0) {
      fprintf(stderr, "Output string mismatch!\n");
      print_details();
      __builtin_trap();
    }
  }

  // Check for out-of-bounds writes
  for (size_t i = dst_size; i < BUFFER_SIZE; ++i) {
    if (impl_dst[i] != SENTINEL) {
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
