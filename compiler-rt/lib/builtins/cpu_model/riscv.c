//=== cpu_model/riscv.c - Update RISC-V Feature Bits Structure -*- C -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "riscv.h"

#if !defined(__riscv)
#error This file is intended only for riscv-based targets
#endif

#define RISCV_FEATURE_BITS_LENGTH 2
struct {
  unsigned length;
  unsigned long long features[RISCV_FEATURE_BITS_LENGTH];
} __riscv_feature_bits __attribute__((visibility("hidden"), nocommon));

struct {
  unsigned mvendorid;
  unsigned long long marchid;
  unsigned long long mimpid;
} __riscv_cpu_model __attribute__((visibility("hidden"), nocommon));

// The formatter wants to re-order these includes, but doing so is incorrect:
// clang-format off
#if defined(__linux__)
#include "riscv/hwprobe.inc"
#else
#include "riscv/unimplemented.inc"
#endif
// clang-format on
