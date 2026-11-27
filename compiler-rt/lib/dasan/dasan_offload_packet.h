//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Offload/host RPC packet for a failed device access.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_OFFLOAD_PACKET_H
#define DASAN_OFFLOAD_PACKET_H

#include <stdint.h>

#include "sanitizer_common/sanitizer_offload_opcodes.h"

enum __dasan_report_kind : uint8_t {
  DASAN_KIND_SPACE = 0,
  DASAN_KIND_SHARED = 1,
  DASAN_KIND_ALLOC = 2,
};

struct __dasan_offload_report {
  uint64_t Addr;       // Address accessed.
  uint64_t Base;       // Base of the object the access was checked against.
  uint64_t AllocSize;  // That object's size, zero once it is freed.
  uint64_t Pc;         // Where the access was, as the wave saw it.
  uint32_t AccessSize;
  uint32_t Block[3];
  uint16_t Thread[3];
  uint8_t Lane;
  uint8_t IsWrite;
  uint8_t Kind;
  uint8_t Recover;
};

static_assert(sizeof(__dasan_offload_report) <= 64,
              "Offload DASan report must fit one RPC packet");

#endif  // DASAN_OFFLOAD_PACKET_H
