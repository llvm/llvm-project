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

#include "RISCVCPUFeatures.inc"

#define SET_SINGLE_RISCV_FEATURE(COND, EXT)                                    \
  if (COND) {                                                                  \
    SET_RISCV_FEATURE(EXT);                                                    \
  }

#define SET_RISCV_FEATURE(EXT) features[EXT##_GROUPID] |= EXT##_BITMASK

// Each platform provides:
//
//   static int initRISCVFeatureBits(unsigned long long features[],
//                                   void *PlatformArgs);
//
// which sets the bits for the detected extensions in features[] (and may
// fill in __riscv_cpu_model). It returns 0 on success, or nonzero if no
// feature information was obtained; __riscv_feature_bits is then left
// untouched and detection is retried on the next call.

// The formatter wants to re-order these includes, but doing so is incorrect:
// clang-format off
#if defined(__linux__)
#include "riscv/hwprobe.inc"
#else
#include "riscv/unimplemented.inc"
#endif
// clang-format on

static int FeaturesBitCached = 0;

static void __init_riscv_feature_bits_ctor(void) CONSTRUCTOR_ATTRIBUTE;

// A constructor function that sets __riscv_feature_bits
// to the right values.  This needs to run only once.  This constructor is given
// the highest priority and it should run before constructors without the
// priority set.  However, it still runs after ifunc initializers and needs to
// be called explicitly there.

static void CONSTRUCTOR_ATTRIBUTE __init_riscv_feature_bits_ctor(void) {
  __init_riscv_feature_bits(0);
}

// PlatformArgs allows the platform to provide pre-computed data and access it
// without extra effort. For example, Linux could pass the vDSO object to avoid
// an extra system call.
void __init_riscv_feature_bits(void *PlatformArgs) {
  unsigned long long features[RISCV_FEATURE_BITS_LENGTH] = {0};
  int i;

  if (FeaturesBitCached)
    return;

  if (initRISCVFeatureBits(features, PlatformArgs))
    return;

  for (i = 0; i < RISCV_FEATURE_BITS_LENGTH; i++)
    __riscv_feature_bits.features[i] = features[i];

  __riscv_feature_bits.length = RISCV_FEATURE_BITS_LENGTH;

  FeaturesBitCached = 1;
}
