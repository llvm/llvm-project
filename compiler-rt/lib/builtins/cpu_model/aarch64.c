//===-- cpu_model/aarch64.c - Support for __cpu_model builtin  ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file is based on LLVM's lib/Support/Host.cpp.
//  It implements __aarch64_have_lse_atomics, __aarch64_cpu_features for
//  AArch64.
//
//===----------------------------------------------------------------------===//

#include "cpu_model.h"

#if !defined(__aarch64__)
#error This file is intended only for aarch64-based targets
#endif

#if __has_include(<sys/ifunc.h>)
#include <sys/ifunc.h>
#else
typedef struct __ifunc_arg_t {
  unsigned long _size;
  unsigned long _hwcap;
  unsigned long _hwcap2;
} __ifunc_arg_t;
#endif // __has_include(<sys/ifunc.h>)

// LSE support detection for out-of-line atomics
// using HWCAP and Auxiliary vector
_Bool __aarch64_have_lse_atomics
    __attribute__((visibility("hidden"), nocommon)) = false;

#if defined(__FreeBSD__)
#include "aarch64/hwcap.inc"
#include "aarch64/lse_atomics/freebsd.inc"
#elif defined(__Fuchsia__)
#include "aarch64/hwcap.inc"
#include "aarch64/lse_atomics/fuchsia.inc"
#elif defined(__ANDROID__)
#include "aarch64/hwcap.inc"
#include "aarch64/lse_atomics/android.inc"
#elif __has_include(<sys/auxv.h>)
#include "aarch64/hwcap.inc"
#include "aarch64/lse_atomics/sysauxv.inc"
#else
// When unimplemented, we leave __aarch64_have_lse_atomics initialized to false.
#endif

#if !defined(DISABLE_AARCH64_FMV)
// CPUFeatures must correspond to the same AArch64 features in
// AArch64TargetParser.h
enum CPUFeatures {
  FEAT_RNG,
  FEAT_FLAGM,
  FEAT_FLAGM2,
  FEAT_FP16FML,
  FEAT_DOTPROD,
  FEAT_SM4,
  FEAT_RDM,
  FEAT_LSE,
  FEAT_FP,
  FEAT_SIMD,
  FEAT_CRC,
  FEAT_SHA1,
  FEAT_SHA2,
  FEAT_SHA3,
  FEAT_AES,
  FEAT_PMULL,
  FEAT_FP16,
  FEAT_DIT,
  FEAT_DPB,
  FEAT_DPB2,
  FEAT_JSCVT,
  FEAT_FCMA,
  FEAT_RCPC,
  FEAT_RCPC2,
  FEAT_FRINTTS,
  FEAT_DGH,
  FEAT_I8MM,
  FEAT_BF16,
  FEAT_EBF16,
  FEAT_RPRES,
  FEAT_SVE,
  FEAT_SVE_BF16,
  FEAT_SVE_EBF16,
  FEAT_SVE_I8MM,
  FEAT_SVE_F32MM,
  FEAT_SVE_F64MM,
  FEAT_SVE2,
  FEAT_SVE_AES,
  FEAT_SVE_PMULL128,
  FEAT_SVE_BITPERM,
  FEAT_SVE_SHA3,
  FEAT_SVE_SM4,
  FEAT_SME,
  FEAT_MEMTAG,
  FEAT_MEMTAG2,
  FEAT_MEMTAG3,
  FEAT_SB,
  FEAT_PREDRES,
  FEAT_SSBS,
  FEAT_SSBS2,
  FEAT_BTI,
  FEAT_LS64,
  FEAT_LS64_V,
  FEAT_LS64_ACCDATA,
  FEAT_WFXT,
  FEAT_SME_F64,
  FEAT_SME_I64,
  FEAT_SME2,
  FEAT_RCPC3,
  FEAT_MAX,
  FEAT_EXT = 62, // Reserved to indicate presence of additional features field
                 // in __aarch64_cpu_features
  FEAT_INIT      // Used as flag of features initialization completion
};

// Architecture features used
// in Function Multi Versioning
struct {
  unsigned long long features;
  // As features grows new fields could be added
} __aarch64_cpu_features __attribute__((visibility("hidden"), nocommon));

// The formatter wants to re-order these includes, but doing so is incorrect:
// clang-format off
#if defined(__APPLE__)
#include "aarch64/fmv/apple.inc"
#elif defined(__FreeBSD__)
#include "aarch64/fmv/mrs.inc"
#include "aarch64/fmv/freebsd.inc"
#elif defined(__Fuchsia__)
#include "aarch64/fmv/mrs.inc"
#include "aarch64/fmv/fuchsia.inc"
#elif defined(__ANDROID__)
#include "aarch64/fmv/mrs.inc"
#include "aarch64/fmv/android.inc"
#elif __has_include(<sys/auxv.h>)
#include "aarch64/fmv/mrs.inc"
#include "aarch64/fmv/sysauxv.inc"
#else
#include "aarch64/fmv/unimplemented.inc"
#endif
// clang-format on

#endif // !defined(DISABLE_AARCH64_FMV)
