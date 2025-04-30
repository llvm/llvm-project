/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Structures to describe the x86 CPU type and CPU features
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "mach.h"

MACHTYPE mach;

#ifdef TARGET_WIN
#define DONT_GENERATE_AVX512  true    /* a temporary restriction */
#else
#define DONT_GENERATE_AVX512  false
#endif

void
set_mach(MACHTYPE *mach, int machtype)
{
  int has_fma3 = 0;
  int has_fma4 = 0;
  memset(&mach->type, 0, sizeof(mach->type));
  memset(&mach->feature, 0, sizeof(mach->feature));
  mach->tpval = machtype;
/*
 *  -tp may not be appropriate for non-x86; so machtype may refer to
 *  machine attriubutes rather than a TP_ value.
 */
#if defined(TARGET_X86)  /* { */
  mach->cachesize = flg.x[32]; /* this applies to all 'machtype's */

  switch (machtype) {
  case TP_ZEN:
    /* AMD Zen microarchitecture, e.g. EPYC and Ryzen processors.
     */
    mach->type[MACH_AMD_ZEN] = 1;
    mach->feature[FEATURE_AVX2] = 1;
    FLANG_FALLTHROUGH;

  case TP_PILEDRIVER:
    /* AMD piledriver
     */
    mach->type[MACH_AMD_PILEDRIVER] = 1;
    mach->feature[FEATURE_FMA3] = 1;
    has_fma3 = 1;
    mach->feature[FEATURE_LD_VMOVUPD] = 1;    /* added on 14 Dec 2015 */
    mach->feature[FEATURE_ST_VMOVUPD] = 1;    /*   "    "    "    "   */
    mach->feature[FEATURE_ST_MOVUPD] = 1;     /*   "    "    "    "   */
    FLANG_FALLTHROUGH;

  case TP_BULLDOZER:
    /* AMD bulldozer
     */
    mach->type[MACH_AMD_BULLDOZER] = 1;
    mach->feature[FEATURE_SSE41] = 1;    /* JHM: added on 2 Feb 2017 */
    mach->feature[FEATURE_SSE42] = 1;    /*  "    "    "    "    "   */
    mach->feature[FEATURE_AVX] = 1;
    if (machtype == TP_PILEDRIVER || machtype == TP_BULLDOZER) {
      mach->feature[FEATURE_SIMD128] = 1;
      mach->feature[FEATURE_FMA4] = 1;
      has_fma4 = 1;
      mach->feature[FEATURE_XOP] = 1;
    }
    mach->feature[FEATURE_ALIGNLOOP8] = 1;
    mach->feature[FEATURE_ALIGNJMP8] = 1;
    FLANG_FALLTHROUGH;

  case TP_ISTANBUL:
    /* AMD instanbul
     */
    mach->type[MACH_AMD_ISTANBUL] = 1;
    FLANG_FALLTHROUGH;

  case TP_SHANGHAI:
    /* AMD shanghai, like greyhound but with a larger cache.
     */
    mach->type[MACH_AMD_SHANGHAI] = 1;
    mach->feature[FEATURE_MULTI_ACCUM] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = (6 * 1024 * 1024);
    FLANG_FALLTHROUGH;

  case TP_GH:
    /* AMD greyhound
     */
    mach->type[MACH_AMD] = 1;
    mach->type[MACH_AMD_HAMMER] = 1;
    mach->type[MACH_AMD_GH] = 1;
    mach->feature[FEATURE_SSE] = 1;
    mach->feature[FEATURE_SSE2] = 1;
    mach->feature[FEATURE_SSE3] = 1;
    mach->feature[FEATURE_MISALIGNEDSSE] = 1;
    mach->feature[FEATURE_LD_MOVUPD] = 1;
    mach->feature[FEATURE_UNROLL_16] = 1;
    mach->feature[FEATURE_DOUBLE_UNROLL] = 1;
    mach->feature[FEATURE_PEEL_SHUFFLE] = 1;
    mach->feature[FEATURE_PREFETCHNTA] = 1;
    mach->feature[FEATURE_PDSHUF] = 1;
    mach->feature[FEATURE_GHLIBS] = 1;
    mach->feature[FEATURE_SSEMISALN] = 1;
    mach->feature[FEATURE_DAZ] = 0;        /* cf. 1 for Intel */
    mach->feature[FEATURE_PREFER_MOVLPD] = 0;
    mach->feature[FEATURE_USE_INC] = 1;    /* cf. 0 for Intel */
    mach->feature[FEATURE_USE_MOVAPD] = 1;
    mach->feature[FEATURE_MERGE_DEPENDENT] = 1;
    mach->feature[FEATURE_SCALAR_NONTEMP] = 1;
    mach->feature[FEATURE_SSE4A] = 1;
    mach->feature[FEATURE_SSEIMAX] = 1;
    mach->feature[FEATURE_ABM] = 1;
    if (machtype == TP_ISTANBUL ||
        machtype == TP_SHANGHAI ||
        machtype == TP_GH)
    {
      if (XBIT(135, 0x400000))
        mach->feature[FEATURE_ALIGNLOOP32] = 1;
      else
        mach->feature[FEATURE_ALIGNLOOP16] = 1;
      mach->feature[FEATURE_ALIGNJMP16] = 1;
    }
    if (mach->cachesize == 0)
      mach->cachesize = (2 * 1024 * 1024);
    break;

  case TP_K8E:
    /* AMD hammer
     */
    mach->feature[FEATURE_SSE3] = 1;
    FLANG_FALLTHROUGH;

  case TP_K8:
    /* AMD hammer
     */
    mach->type[MACH_AMD] = 1;
    mach->type[MACH_AMD_HAMMER] = 1;
    mach->feature[FEATURE_SSE] = 1;
    mach->feature[FEATURE_SSE2] = 1;
    mach->feature[FEATURE_DAZ] = 0;    /* cf. 1 for Intel */
    mach->feature[FEATURE_PREFER_MOVLPD] = 1;
    mach->feature[FEATURE_USE_INC] = 1;
    mach->feature[FEATURE_ALIGNLOOP16] = 1;
    mach->feature[FEATURE_ALIGNJMP16] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = (1024 * 1024);
    break;

  case TP_LARRABEE:
    mach->type[MACH_INTEL] = 1;
    mach->type[MACH_INTEL_PENTIUM4] = 1;
    mach->type[MACH_INTEL_LARRABEE] = 1;
    mach->feature[FEATURE_SSE] = 1;
    mach->feature[FEATURE_SSE2] = 1;
    mach->feature[FEATURE_SSE3] = 1;
    mach->feature[FEATURE_SSE41] = 1;
    mach->feature[FEATURE_SSE42] = 1;
    mach->feature[FEATURE_USE_INC] = 0;
    mach->feature[FEATURE_LD_MOVUPD] = 1;
    mach->feature[FEATURE_USE_MOVAPD] = 1;
    mach->feature[FEATURE_MNI] = 1;
    mach->feature[FEATURE_DAZ] = 1;
    mach->feature[FEATURE_SSEIMAX] = 1;
    mach->feature[FEATURE_SSEPMAX] = 1;
    mach->feature[FEATURE_LRBNI] = 1;
    mach->feature[FEATURE_NOPREFETCH] = 1;
    mach->feature[FEATURE_ALIGNLOOP8] = 1;
    mach->feature[FEATURE_ALIGNJMP8] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = 262144;
    break;

  case TP_SKYLAKE:
    if (! DONT_GENERATE_AVX512) {
      mach->type[MACH_INTEL_SKYLAKE] = 1;
      mach->feature[FEATURE_AVX512VL] = 1;
    }
    FLANG_FALLTHROUGH;

  case TP_KNIGHTS_LANDING:
    if (! DONT_GENERATE_AVX512) {
      if (machtype == TP_KNIGHTS_LANDING) {
        mach->type[MACH_INTEL_KNIGHTS_LANDING] = 1;
      }
      mach->feature[FEATURE_AVX512F] = 1;
    }
    FLANG_FALLTHROUGH;

  case TP_HASWELL:
    mach->type[MACH_INTEL_HASWELL] = 1;
    mach->feature[FEATURE_AVX2] = 1;
    mach->feature[FEATURE_FMA3] = 1;
    has_fma3 = 1;
    mach->feature[FEATURE_LD_VMOVUPD] = 1;
    mach->feature[FEATURE_ST_VMOVUPD] = 1;
    FLANG_FALLTHROUGH;

  case TP_IVYBRIDGE:
  case TP_SANDYBRIDGE:
    mach->type[MACH_INTEL_SANDYBRIDGE] = 1;
    mach->feature[FEATURE_AVX] = 1;
    mach->feature[FEATURE_ST_MOVUPD] = 1;
    mach->feature[FEATURE_MULTI_ACCUM] = 1;
    FLANG_FALLTHROUGH;

  case TP_NEHALEM:
    mach->type[MACH_INTEL_NEHALEM] = 1;
    mach->feature[FEATURE_SSE42] = 1;
    mach->feature[FEATURE_LD_MOVUPD] = 1;
    mach->feature[FEATURE_SSEIMAX] = 1;
    mach->feature[FEATURE_SSEPMAX] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = (8 * 1024 * 1024);
    FLANG_FALLTHROUGH;

  case TP_PENRYN:
    mach->type[MACH_INTEL_PENRYN] = 1;
    mach->feature[FEATURE_SSE41] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = (6 * 1024 * 1024);
    FLANG_FALLTHROUGH;

  case TP_CORE2:
    mach->type[MACH_INTEL_CORE2] = 1;
    mach->feature[FEATURE_SSE3] = 1;
    mach->feature[FEATURE_MNI] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = (4 * 1024 * 1024);
    FLANG_FALLTHROUGH;

  case TP_P7:
    /* Intel P7 Pentium IV
     */
    mach->type[MACH_INTEL] = 1;
    mach->type[MACH_INTEL_PENTIUM4] = 1;
    mach->feature[FEATURE_SSE] = 1;
    mach->feature[FEATURE_SSE2] = 1;
    mach->feature[FEATURE_USE_INC] = 0;    /* cf. 1 for AMD */
    mach->feature[FEATURE_USE_MOVAPD] = 1;
    mach->feature[FEATURE_DAZ] = 1;        /* cf. 0 for AMD */
    mach->feature[FEATURE_ALIGNLOOP8] = 1;
    mach->feature[FEATURE_ALIGNJMP8] = 1;
    if (XBIT(80, 0x4000000)) {
      mach->feature[FEATURE_SSE3] = 1;
    }
    if (machtype == TP_PENRYN || machtype == TP_CORE2 || machtype == TP_P7) {
      mach->feature[FEATURE_NOPREFETCH] = 1;
    }
    if (mach->cachesize == 0)
      mach->cachesize = (1024 * 1024);
    break;

#ifdef TARGET_X8664
  case TP_PY:
  case TP_PX:
    /* we know all x86-64 have at least SSE and SSE2 */
    /* more or less the same as p7 without sse3, without -Mdaz */
    mach->type[MACH_INTEL] = 1;
    mach->type[MACH_INTEL_PENTIUM4] = 1;
    mach->feature[FEATURE_SSE] = 1;
    mach->feature[FEATURE_SSE2] = 1;
    mach->feature[FEATURE_USE_INC] = 0;
    mach->feature[FEATURE_USE_MOVAPD] = 1;
    mach->feature[FEATURE_NOPREFETCH] = 1;
    mach->feature[FEATURE_ALIGNLOOP8] = 1;
    mach->feature[FEATURE_ALIGNJMP8] = 1;
    if (mach->cachesize == 0)
      mach->cachesize = (1024 * 1024);
    break;
#else
  case TP_PY: /* for 32-bit, treat like generic */
  case TP_PX: /*   "     "     "     "     "    */
#endif
  default:
    mach->type[MACH_GENERIC] = 1;
    mach->feature[FEATURE_SSE] = 1;
    mach->feature[FEATURE_SSE2] = 1;
    mach->feature[FEATURE_USE_INC] = 0;
    mach->feature[FEATURE_NOPREFETCH] = 1;
    mach->feature[FEATURE_ALIGNLOOP4] = 1;
    mach->feature[FEATURE_ALIGNJMP8] = 1;
    if (XBIT(129, 4))
      mach->feature[FEATURE_DAZ] = 1;
    mach->tpval = TP_PX;
    if (mach->cachesize == 0)
      mach->cachesize = 262144;
    break;
  }    /* end switch (machtype) */
#endif /* defined(TARGET_X86) } */

#if defined(TARGET_LLVM_ARM)
  mach->feature[FEATURE_SCALAR_NEON] = 1;
  mach->feature[FEATURE_NEON] = 1;
  mach->feature[FEATURE_FMA] = 1;
  has_fma3 = 1;
#elif defined(TARGET_LLVM_POWER)
  mach->feature[FEATURE_SCALAR_VSX] = 1;
  mach->feature[FEATURE_VSX] = 1;
  mach->feature[FEATURE_FMA] = 1;
  has_fma3 = 1;
#elif defined(X86_64)
  /* new cg or 64-bit cg */
  mach->feature[FEATURE_SCALAR_SSE] = 1;
#endif

  /* override machine-specific settings of DAZ */
  if (XBIT(129, 4))
    mach->feature[FEATURE_DAZ] = 1;
  else if (XBIT(129, 0x400))
    mach->feature[FEATURE_DAZ] = 0;
  if (XBIT(135, 0x20))
    mach->feature[FEATURE_USE_MOVAPD] = 1;

  /* -Mnoprefetch */
  if (XBIT(39, 1))
    mach->feature[FEATURE_NOPREFETCH] = 1;

  /* -Mvect=simd:128 */
  if (XBIT(56, 0x40))
    mach->feature[FEATURE_SIMD128] = 1;
  /* -Mvect=simd:256 or -Mvect=simd:512 */
  if (XBIT(56, 0x100) || XBIT(56, 0x800))
    mach->feature[FEATURE_SIMD128] = 0;

  /* align 16 before loops */
  if (XBIT(135, 0x2000) && !XBIT(135, 0x4000)) {
    mach->feature[FEATURE_ALIGNLOOP4] = 0;
    mach->feature[FEATURE_ALIGNLOOP8] = 0;
    mach->feature[FEATURE_ALIGNLOOP16] = 1;
    mach->feature[FEATURE_ALIGNLOOP32] = 0;
  }
  /* align 8 before loops */
  if (XBIT(135, 0x2000) && XBIT(135, 0x4000)) {
    mach->feature[FEATURE_ALIGNLOOP4] = 0;
    mach->feature[FEATURE_ALIGNLOOP8] = 1;
    mach->feature[FEATURE_ALIGNLOOP16] = 0;
    mach->feature[FEATURE_ALIGNLOOP32] = 0;
  }
  /* no align before loops */
  if (XBIT(135, 0x8000) || (XBIT(135, 0x4000) && !XBIT(135, 0x2000))) {
    mach->feature[FEATURE_ALIGNLOOP4] = 0;
    mach->feature[FEATURE_ALIGNLOOP8] = 0;
    mach->feature[FEATURE_ALIGNLOOP16] = 0;
    mach->feature[FEATURE_ALIGNLOOP32] = 0;
  }
  /* align 16 after jmp */
  if (XBIT(135, 0x10000) && XBIT(135, 0x8000)) {
    mach->feature[FEATURE_ALIGNJMP8] = 0;
    mach->feature[FEATURE_ALIGNJMP16] = 1;
  }
  /* align 8 after jmp */
  if (!XBIT(135, 0x10000) && XBIT(135, 0x8000)) {
    mach->feature[FEATURE_ALIGNJMP8] = 1;
    mach->feature[FEATURE_ALIGNJMP16] = 0;
  }
  /* no align after jmp */
  if (XBIT(135, 0x10000) && !XBIT(135, 0x8000)) {
    mach->feature[FEATURE_ALIGNJMP8] = 0;
    mach->feature[FEATURE_ALIGNJMP16] = 0;
  }

  /* override feature settings */
  if (XBIT(171, 1))
    mach->feature[FEATURE_SCALAR_SSE] = 0;
  else if (XBIT(172, 1))
    mach->feature[FEATURE_SCALAR_SSE] = 1;
  if (XBIT(171, 2))
    mach->feature[FEATURE_SSE] = 0;
  else if (XBIT(172, 2))
    mach->feature[FEATURE_SSE] = 1;
  if (XBIT(171, 4))
    mach->feature[FEATURE_SSE2] = 0;
  else if (XBIT(172, 4))
    mach->feature[FEATURE_SSE2] = 1;
  if (XBIT(171, 8))
    mach->feature[FEATURE_SSE3] = 0;
  else if (XBIT(172, 8))
    mach->feature[FEATURE_SSE3] = 1;
  if (XBIT(171, 0x10))
    mach->feature[FEATURE_SSE41] = 0;
  else if (XBIT(172, 0x10))
    mach->feature[FEATURE_SSE41] = 1;
  if (XBIT(171, 0x20))
    mach->feature[FEATURE_SSE42] = 0;
  else if (XBIT(172, 0x20))
    mach->feature[FEATURE_SSE42] = 1;
  if (XBIT(171, 0x40))
    mach->feature[FEATURE_SSE4A] = 0;
  else if (XBIT(172, 0x40))
    mach->feature[FEATURE_SSE4A] = 1;
  if (XBIT(171, 0x80))
    mach->feature[FEATURE_SSE5] = 0;
  else if (XBIT(172, 0x80))
    mach->feature[FEATURE_SSE5] = 1;
  if (XBIT(171, 0x100))
    mach->feature[FEATURE_MNI] = 0;
  else if (XBIT(172, 0x100))
    mach->feature[FEATURE_MNI] = 1;
  if (XBIT(171, 0x200))
    mach->feature[FEATURE_DAZ] = 0;
  else if (XBIT(172, 0x200))
    mach->feature[FEATURE_DAZ] = 1;
  if (XBIT(171, 0x400))
    mach->feature[FEATURE_PREFER_MOVLPD] = 0;
  else if (XBIT(172, 0x400))
    mach->feature[FEATURE_PREFER_MOVLPD] = 1;
  if (XBIT(171, 0x800))
    mach->feature[FEATURE_USE_INC] = 0;
  else if (XBIT(172, 0x800))
    mach->feature[FEATURE_USE_INC] = 1;
  if (XBIT(171, 0x1000))
    mach->feature[FEATURE_USE_MOVAPD] = 0;
  else if (XBIT(172, 0x1000))
    mach->feature[FEATURE_USE_MOVAPD] = 1;
  if (XBIT(171, 0x2000))
    mach->feature[FEATURE_MERGE_DEPENDENT] = 0;
  else if (XBIT(172, 0x2000))
    mach->feature[FEATURE_MERGE_DEPENDENT] = 1;
  if (XBIT(171, 0x4000))
    mach->feature[FEATURE_SCALAR_NONTEMP] = 0;
  else if (XBIT(172, 0x4000))
    mach->feature[FEATURE_SCALAR_NONTEMP] = 1;
  if (XBIT(171, 0x8000))
    mach->feature[FEATURE_SSEIMAX] = 0;
  else if (XBIT(172, 0x8000))
    mach->feature[FEATURE_SSEIMAX] = 1;
  if (XBIT(171, 0x10000))
    mach->feature[FEATURE_MISALIGNEDSSE] = 0;
  else if (XBIT(172, 0x10000))
    mach->feature[FEATURE_MISALIGNEDSSE] = 1;
  if (XBIT(171, 0x20000))
    mach->feature[FEATURE_LD_MOVUPD] = 0;
  else if (XBIT(172, 0x20000))
    mach->feature[FEATURE_LD_MOVUPD] = 1;
  if (XBIT(171, 0x40000))
    mach->feature[FEATURE_ST_MOVUPD] = 0;
  else if (XBIT(172, 0x40000))
    mach->feature[FEATURE_ST_MOVUPD] = 1;
  if (XBIT(171, 0x80000))
    mach->feature[FEATURE_UNROLL_16] = 0;
  else if (XBIT(172, 0x80000))
    mach->feature[FEATURE_UNROLL_16] = 1;
  if (XBIT(171, 0x100000))
    mach->feature[FEATURE_DOUBLE_UNROLL] = 0;
  else if (XBIT(172, 0x100000))
    mach->feature[FEATURE_DOUBLE_UNROLL] = 1;
  if (XBIT(171, 0x200000))
    mach->feature[FEATURE_PEEL_SHUFFLE] = 0;
  else if (XBIT(172, 0x200000))
    mach->feature[FEATURE_PEEL_SHUFFLE] = 1;
  if (XBIT(171, 0x400000))
    mach->feature[FEATURE_PREFETCHNTA] = 0;
  else if (XBIT(172, 0x400000))
    mach->feature[FEATURE_PREFETCHNTA] = 1;
  if (XBIT(171, 0x800000))
    mach->feature[FEATURE_PDSHUF] = 0;
  else if (XBIT(172, 0x800000))
    mach->feature[FEATURE_PDSHUF] = 1;
  if (XBIT(171, 0x1000000))
    mach->feature[FEATURE_SSEPMAX] = 0;
  else if (XBIT(172, 0x1000000))
    mach->feature[FEATURE_SSEPMAX] = 1;
  if (XBIT(171, 0x2000000))
    mach->feature[FEATURE_GHLIBS] = 0;
  else if (XBIT(172, 0x2000000))
    mach->feature[FEATURE_GHLIBS] = 1;
  if (XBIT(171, 0x4000000))
    mach->feature[FEATURE_SSEMISALN] = 0;
  else if (XBIT(172, 0x4000000))
    mach->feature[FEATURE_SSEMISALN] = 1;
  if (XBIT(171, 0x8000000))
    mach->feature[FEATURE_ABM] = 0;
  else if (XBIT(172, 0x8000000))
    mach->feature[FEATURE_ABM] = 1;
  if (XBIT(171, 0x10000000))
    mach->feature[FEATURE_AVX] = 0;
  else if (XBIT(172, 0x10000000))
    mach->feature[FEATURE_AVX] = 1;
  if (XBIT(171, 0x20000000))
    mach->feature[FEATURE_LRBNI] = 0;
  else if (XBIT(172, 0x20000000))
    mach->feature[FEATURE_LRBNI] = 1;
  if (has_fma4) {
    if (XBIT(171, 0x40000000))
      mach->feature[FEATURE_FMA4] = 0;
    else if (XBIT(172, 0x40000000))
      mach->feature[FEATURE_FMA4] = 1;
  }
  if (XBIT(171, 0x80000000))
    mach->feature[FEATURE_XOP] = 0;
  else if (XBIT(172, 0x80000000))
    mach->feature[FEATURE_XOP] = 1;
  if (has_fma3) {
    if (XBIT(178, 0x01))
      mach->feature[FEATURE_FMA3] = 0;
    else if (XBIT(179, 0x01))
      mach->feature[FEATURE_FMA3] = 1;
  }
  if (XBIT(178, 0x02))
    mach->feature[FEATURE_MULTI_ACCUM] = 0;
  else if (XBIT(179, 0x02))
    mach->feature[FEATURE_MULTI_ACCUM] = 1;
  if (XBIT(178, 0x04))
    mach->feature[FEATURE_SIMD128] = 0;
  else if (XBIT(179, 0x04))
    mach->feature[FEATURE_SIMD128] = 1;
  if (XBIT(178, 0x08))
    mach->feature[FEATURE_NOPREFETCH] = 0;
  else if (XBIT(179, 0x08))
    mach->feature[FEATURE_NOPREFETCH] = 1;
  if (XBIT(178, 0x10))
    mach->feature[FEATURE_ALIGNLOOP4] = 0;
  else if (XBIT(179, 0x10))
    mach->feature[FEATURE_ALIGNLOOP4] = 1;
  if (XBIT(178, 0x20))
    mach->feature[FEATURE_ALIGNLOOP8] = 0;
  else if (XBIT(179, 0x20))
    mach->feature[FEATURE_ALIGNLOOP8] = 1;
  if (XBIT(178, 0x40))
    mach->feature[FEATURE_ALIGNLOOP16] = 0;
  else if (XBIT(179, 0x40))
    mach->feature[FEATURE_ALIGNLOOP16] = 1;
  if (XBIT(178, 0x80))
    mach->feature[FEATURE_ALIGNLOOP32] = 0;
  else if (XBIT(179, 0x80))
    mach->feature[FEATURE_ALIGNLOOP32] = 1;
  if (XBIT(178, 0x100))
    mach->feature[FEATURE_LD_VMOVUPD] = 0;
  else if (XBIT(179, 0x100))
    mach->feature[FEATURE_LD_VMOVUPD] = 1;
  if (XBIT(178, 0x200))
    mach->feature[FEATURE_ST_VMOVUPD] = 0;
  else if (XBIT(179, 0x200))
    mach->feature[FEATURE_ST_VMOVUPD] = 1;
  if (XBIT(178, 0x400))
    mach->feature[FEATURE_AVX2] = 0;
  else if (XBIT(179, 0x400))
    mach->feature[FEATURE_AVX2] = 1;
  if (XBIT(178, 0x800))
    mach->feature[FEATURE_AVX512F] = 0;
  else if (XBIT(179, 0x800))
    mach->feature[FEATURE_AVX512F] = 1;
  if (XBIT(178, 0x2000))
    mach->feature[FEATURE_AVX512VL] = 0;
  else if (XBIT(179, 0x2000))
    mach->feature[FEATURE_AVX512VL] = 1;

} /* set_mach */

/* take intersection of all mach-> features */
static MACHTYPE mach_intersect;
void
init_mach_intersect()
{
  int i;
  mach_intersect.tpval = 0;
  /* take minimum of all cache sizes */
  mach_intersect.cachesize = 0;
  for (i = 0; i < MACH_NUMBER; ++i)
    mach_intersect.type[i] = 1;
  for (i = 0; i < FEATURE_NUMBER; ++i)
    mach_intersect.feature[i] = 1;
} /* init_machintersect */

void
intersect_mach_intersect(MACHTYPE *mach)
{
  int i;
  if (mach_intersect.cachesize == 0 ||
      (mach->cachesize && mach->cachesize > mach_intersect.cachesize))
    mach_intersect.cachesize = mach->cachesize;
  for (i = 0; i < MACH_NUMBER; ++i) {
    if (!mach->type[i])
      mach_intersect.type[i] = 0;
  }
  for (i = 0; i < FEATURE_NUMBER; ++i) {
    if (!mach->feature[i])
      mach_intersect.feature[i] = 0;
  }
} /* intersect_mach_intersect */

void
copy_mach_intersect(MACHTYPE *mach)
{
  int i;
  mach->cachesize = mach_intersect.cachesize;
  for (i = 0; i < MACH_NUMBER; ++i)
    mach->type[i] = mach_intersect.type[i];
  for (i = 0; i < FEATURE_NUMBER; ++i)
    mach->feature[i] = mach_intersect.feature[i];
} /* copy_mach_intersect */

int
machvalue(const char *thistpname)
{
#ifdef TARGET_X8664
  if (strcmp(thistpname, "amd64") == 0)
    return TP_K8;
  if (strcmp(thistpname, "amd64e") == 0)
    return TP_K8E;
#endif
  if (strcmp(thistpname, "athlon") == 0)
    return TP_K8;
  if (strcmp(thistpname, "bulldozer") == 0)
    return TP_BULLDOZER;
  if (strncmp(thistpname, "core2", 5) == 0)
    return TP_CORE2;
  if (strncmp(thistpname, "gh", 2) == 0)
    return TP_GH;
  if (strncmp(thistpname, "hammer", 6) == 0)
    return TP_K8;
  if (strncmp(thistpname, "haswell", 9) == 0)
    return TP_HASWELL;
  if (strncmp(thistpname, "istanbul", 8) == 0)
    return TP_ISTANBUL;
  if (strncmp(thistpname, "ivybridge", 9) == 0)
    return TP_IVYBRIDGE;
  if (strcmp(thistpname, "k8") == 0)
    return TP_K8;
  if (strncmp(thistpname, "k8", 2) == 0 &&
      thistpname[strlen(thistpname) - 1] == 'e')
    return TP_K8E;
  if (strncmp(thistpname, "k8", 2) == 0)
    return TP_K8;
  if (strncmp(thistpname, "knl", 3) == 0)
    return TP_KNIGHTS_LANDING;
  if (strncmp(thistpname, "nehalem", 7) == 0)
    return TP_NEHALEM;
  if (strncmp(thistpname, "p7", 2) == 0)
    return TP_P7;
  if (strncmp(thistpname, "penryn", 6) == 0)
    return TP_PENRYN;
  if (strcmp(thistpname, "piledriver") == 0)
    return TP_PILEDRIVER;
  if (strncmp(thistpname, "px", 2) == 0)
    return TP_PX;
  if (strncmp(thistpname, "py", 2) == 0)
    return TP_PY;
  if (strncmp(thistpname, "sandybridge", 11) == 0)
    return TP_SANDYBRIDGE;
  if (strncmp(thistpname, "shanghai", 8) == 0)
    return TP_SHANGHAI;
  if (strncmp(thistpname, "skylake", 7) == 0)
    return TP_SKYLAKE;
  if (strncmp(thistpname, "zen", 3) == 0)
    return TP_ZEN;
  return 0;
} /* machvalue */

void
set_tp(const char *thistpname)
{
  if (flg.tpcount <= TPNVERSION) {
    int n;
    n = machvalue(thistpname);
    if (n <= 0) {
      interr("Unexpected value for -tp switch", 0, ERR_Fatal);
    } else {
      if (flg.tpcount == 0) {
        flg.tpvalue[flg.tpcount] = n;
        ++flg.tpcount;
      } else {
      if (n < flg.tpvalue[0])
        flg.tpvalue[0] = n;
      }
    }
  }
} /* set_tp */

void
check_tp(bool skip)
{
}

#if DEBUG
const char *
sxtp(int tp)
{
  switch (tp) {
  case TP_PY:
    return "py";
  case TP_PX:
    return "px";
  case TP_P5:
    return "p5";
  case TP_ATHLON:
    return "athlon";
  case TP_P6:
    return "p6";
  case TP_ATHLON_XP:
    return "athlon_xp";
  case TP_PIII:
    return "piii";
  case TP_K8:
    return "k8";
  case TP_P7:
    return "p7";
  case TP_K8E:
    return "k8e";
  case TP_PIV:
    return "piv";
  case TP_GH:
    return "gh";
  case TP_CORE2:
    return "core2";
  case TP_PENRYN:
    return "penryn";
  case TP_SHANGHAI:
    return "shanghai";
  case TP_ISTANBUL:
    return "istanbul";
  case TP_NEHALEM:
    return "nehalem";
  case TP_BULLDOZER:
    return "bulldozer";
  case TP_SANDYBRIDGE:
    return "sandybridge";
  case TP_IVYBRIDGE:
    return "ivybridge";
  case TP_HASWELL:
    return "haswell";
  case TP_LARRABEE:
    return "larrabee";
  case TP_PILEDRIVER:
    return "piledriver";
  case TP_KNIGHTS_LANDING:
    return "knl";
  case TP_SKYLAKE:
    return "skylake";
  case TP_ZEN:
    return "zen";
  default:
    return "??";
  }
} /* sxtp */

const char *
sxtype(int m)
{
  switch (m) {
  case MACH_GENERIC:
    return "mach_generic";
  case MACH_INTEL:
    return "mach_intel";
  case MACH_INTEL_PENTIUM4:
    return "mach_pentium4";
  case MACH_INTEL_CORE2:
    return "mach_core2";
  case MACH_INTEL_PENRYN:
    return "mach_penryn";
  case MACH_INTEL_NEHALEM:
    return "mach_nehalem";
  case MACH_INTEL_SANDYBRIDGE:
    return "mach_sandybridge";
  case MACH_INTEL_HASWELL:
    return "mach_haswell";
  case MACH_INTEL_KNIGHTS_LANDING:
    return "mach_knl";
  case MACH_INTEL_SKYLAKE:
    return "mach_skylake";
  case MACH_INTEL_LARRABEE:
    return "mach_larrabee";
  case MACH_AMD:
    return "mach_amd";
  case MACH_AMD_ATHLON:
    return "mach_athlon";
  case MACH_AMD_ATHLON_XP:
    return "mach_athlon_xp";
  case MACH_AMD_HAMMER:
    return "mach_hammer";
  case MACH_AMD_GH:
    return "mach_gh";
  case MACH_AMD_SHANGHAI:
    return "mach_shanghai";
  case MACH_AMD_ISTANBUL:
    return "mach_istanbul";
  case MACH_AMD_BULLDOZER:
    return "mach_bulldozer";
  case MACH_AMD_PILEDRIVER:
    return "mach_piledriver";
  case MACH_AMD_ZEN:
    return "mach_zen";
  default:
    return "??";
  }
} /* sxtype */

const char *
sxfeature(int f)
{
  switch (f) {
  case FEATURE_SCALAR_SSE:
    return "feature_scalar_sse";
  case FEATURE_SSE:
    return "feature_sse";
  case FEATURE_SSE2:
    return "feature_sse2";
  case FEATURE_SSE3:
    return "feature_sse3";
  case FEATURE_SSE41:
    return "feature_sse41";
  case FEATURE_SSE42:
    return "feature_sse42";
  case FEATURE_SSE4A:
    return "feature_sse4a";
  case FEATURE_SSE5:
    return "feature_sse5";
  case FEATURE_MNI:
    return "feature_mni";
  case FEATURE_DAZ:
    return "feature_daz";
  case FEATURE_PREFER_MOVLPD:
    return "feature_prever_movlpd";
  case FEATURE_USE_INC:
    return "feature_use_inc";
  case FEATURE_USE_MOVAPD:
    return "feature_use_movapd";
  case FEATURE_MERGE_DEPENDENT:
    return "feature_merge_dependent";
  case FEATURE_SCALAR_NONTEMP:
    return "feature_scalar_nontemp";
  case FEATURE_SSEIMAX:
    return "feature_sseimax";
  case FEATURE_MISALIGNEDSSE:
    return "feature_misalignedsse";
  case FEATURE_LD_MOVUPD:
    return "feature_ld_movupd";
  case FEATURE_ST_MOVUPD:
    return "feature_st_movupd";
  case FEATURE_UNROLL_16:
    return "feature_unroll_16";
  case FEATURE_DOUBLE_UNROLL:
    return "feature_double_unroll";
  case FEATURE_PEEL_SHUFFLE:
    return "feature_peel_shuffle";
  case FEATURE_PREFETCHNTA:
    return "feature_prefetchnta";
  case FEATURE_PDSHUF:
    return "feature_pdshuf";
  case FEATURE_SSEPMAX:
    return "feature_ssepmax";
  case FEATURE_GHLIBS:
    return "feature_ghlibs";
  case FEATURE_SSEMISALN:
    return "feature_ssemisaln";
  case FEATURE_ABM:
    return "feature_abm";
  case FEATURE_AVX:
    return "feature_avx";
  case FEATURE_LRBNI:
    return "feature_lrbni";
  case FEATURE_FMA4:
    return "feature_fma4";
  case FEATURE_XOP:
    return "feature_xop";
  case FEATURE_FMA3:
    return "feature_fma3";
  case FEATURE_MULTI_ACCUM:
    return "feature_multi_accum";
  case FEATURE_SIMD128:
    return "feature_simd128";
  case FEATURE_NOPREFETCH:
    return "feature_noprefetch";
  case FEATURE_ALIGNLOOP4:
    return "feature_alignloop4";
  case FEATURE_ALIGNLOOP8:
    return "feature_alignloop8";
  case FEATURE_ALIGNLOOP16:
    return "feature_alignloop16";
  case FEATURE_ALIGNLOOP32:
    return "feature_alignloop32";
  case FEATURE_ALIGNJMP8:
    return "feature_alignjmp8";
  case FEATURE_ALIGNJMP16:
    return "feature_alignjmp16";
  case FEATURE_LD_VMOVUPD:
    return "feature_ld_vmovupd";
  case FEATURE_ST_VMOVUPD:
    return "feature_st_vmovupd";
  case FEATURE_AVX2:
    return "feature_avx2";
  case FEATURE_AVX512F:
    return "feature_avx512f";
  case FEATURE_AVX512VL:
    return "feature_avx512vl";
  default:
    return "??";
  }
} /* sxfeature */

void
_dumpmach(MACHTYPE *mach)
{
  FILE *dfile;
  int m, f;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "%d=tpval=%s\n", mach->tpval, sxtp(mach->tpval));
  for (m = 0; m < MACH_NUMBER; ++m) {
    if (mach->type[m]) {
      fprintf(dfile, "%d=type[%2d]=%s\n", mach->type[m], m, sxtype(m));
    }
  }

  for (f = 0; f < FEATURE_NUMBER; ++f) {
    if (mach->feature[f]) {
      fprintf(dfile, "%d=feature[%2d]=%s\n", mach->feature[f], f, sxfeature(f));
    }
  }

  fprintf(dfile, "%ld=cachesize\n", mach->cachesize);
} /* _dumpmach */

void
dumpmach()
{
  _dumpmach(&mach);
} /* dumpmach */
#endif
