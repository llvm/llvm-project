//===-- ubsan_device_packet.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device/host RPC packet for UBSan reports.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_DEVICE_PACKET_H
#define UBSAN_DEVICE_PACKET_H

#include <stdint.h>

#define UBSAN_DEVICE_REPORT_OPCODE (('s' << 24) | 0)

enum __ubsan_report_kind : uint8_t {
#define UBSAN_DEVICE_HANDLER(kind, ...) UBSAN_DEVICE_##kind,
#define UBSAN_DEVICE_HANDLER_NORETURN(kind, ...) UBSAN_DEVICE_##kind,
#include "ubsan_device_checks.inc"
  UBSAN_DEVICE_KIND_COUNT
};

// A single device packet to initiate the report in the UBSan host runtime.
struct __ubsan_device_report {
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

static_assert(sizeof(__ubsan_device_report) == 64,
              "Device UBSan report must fit one RPC packet");

#endif // UBSAN_DEVICE_PACKET_H
