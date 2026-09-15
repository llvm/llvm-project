//===-- ubsan_offload_packet.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offload/host RPC packet for UBSan reports.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_OFFLOAD_PACKET_H
#define UBSAN_OFFLOAD_PACKET_H

#include <stdint.h>

#define UBSAN_OFFLOAD_REPORT_OPCODE (('s' << 24) | 0)

enum __ubsan_report_kind : uint8_t {
#define UBSAN_OFFLOAD_HANDLER(kind, ...) UBSAN_OFFLOAD_##kind,
#define UBSAN_OFFLOAD_HANDLER_NORETURN(kind, ...) UBSAN_OFFLOAD_##kind,
#include "ubsan_offload_checks.inc"
  UBSAN_OFFLOAD_KIND_COUNT
};

// A single offload packet to initiate the report in the UBSan host runtime.
struct __ubsan_offload_report {
  uint64_t pc;
  uint64_t data;
  uint64_t val0;
  uint8_t reserved0[8];
  uint64_t val1;
  uint8_t reserved1[8];
  uint64_t val2;
  uint8_t kind;
  uint8_t fatal;
  uint8_t reserved2[3];
};

static_assert(sizeof(__ubsan_offload_report) == 64,
              "Offload UBSan report must fit one RPC packet");

#endif // UBSAN_OFFLOAD_PACKET_H
