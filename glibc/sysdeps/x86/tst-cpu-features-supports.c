/* Test CPU feature data against __builtin_cpu_supports.
   This file is part of the GNU C Library.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/platform/x86.h>
#include <stdio.h>

int
check_supports (int supports, int active, const char *supports_name,
		const char *name)
{
  printf ("Checking %s:\n", name);
  printf ("  %s: %d\n", name, active);
  printf ("  __builtin_cpu_supports (%s): %d\n",
	  supports_name, supports);

  if ((supports != 0) != (active != 0))
    {
      printf (" *** failure ***\n");
      return 1;
    }

  return 0;
}

#define CHECK_FEATURE_ACTIVE(str, name) \
  check_supports (__builtin_cpu_supports (#str), \
		  CPU_FEATURE_ACTIVE (name), \
		  #str, "CPU_FEATURE_ACTIVE (" #name ")");

#define CHECK_FEATURE_PRESENT(str, name) \
  check_supports (__builtin_cpu_supports (#str), \
		  CPU_FEATURE_PRESENT (name), \
		  #str, "CPU_FEATURE_PRESENT (" #name ")");

static int
do_test (int argc, char **argv)
{
  int fails = 0;

#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (adx, ADX);
#endif
#if __GNUC_PREREQ (6, 0)
  fails += CHECK_FEATURE_ACTIVE (aes, AES);
#endif
#if __GNUC_PREREQ (11, 1)
  fails += CHECK_FEATURE_ACTIVE (amx-bf16, AMX_BF16);
  fails += CHECK_FEATURE_ACTIVE (amx-int8, AMX_INT8);
  fails += CHECK_FEATURE_ACTIVE (amx-tile, AMX_TILE);
#endif
  fails += CHECK_FEATURE_ACTIVE (avx, AVX);
  fails += CHECK_FEATURE_ACTIVE (avx2, AVX2);
#if __GNUC_PREREQ (7, 0)
  fails += CHECK_FEATURE_ACTIVE (avx5124fmaps, AVX512_4FMAPS);
  fails += CHECK_FEATURE_ACTIVE (avx5124vnniw, AVX512_4VNNIW);
#endif
#if __GNUC_PREREQ (10, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512bf16, AVX512_BF16);
#endif
#if __GNUC_PREREQ (8, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512bitalg, AVX512_BITALG);
#endif
#if __GNUC_PREREQ (6, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512ifma, AVX512_IFMA);
  fails += CHECK_FEATURE_ACTIVE (avx512vbmi, AVX512_VBMI);
#endif
#if __GNUC_PREREQ (8, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512vbmi2, AVX512_VBMI2);
  fails += CHECK_FEATURE_ACTIVE (avx512vnni, AVX512_VNNI);
#endif
#if __GNUC_PREREQ (10, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512vp2intersect, AVX512_VP2INTERSECT);
#endif
#if __GNUC_PREREQ (7, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512vpopcntdq, AVX512_VPOPCNTDQ);
#endif
#if __GNUC_PREREQ (6, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512bw, AVX512BW);
  fails += CHECK_FEATURE_ACTIVE (avx512cd, AVX512CD);
  fails += CHECK_FEATURE_ACTIVE (avx512er, AVX512ER);
  fails += CHECK_FEATURE_ACTIVE (avx512dq, AVX512DQ);
#endif
#if __GNUC_PREREQ (5, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512f, AVX512F);
#endif
#if __GNUC_PREREQ (6, 0)
  fails += CHECK_FEATURE_ACTIVE (avx512pf, AVX512PF);
  fails += CHECK_FEATURE_ACTIVE (avx512vl, AVX512VL);
#endif
#if __GNUC_PREREQ (5, 0)
  fails += CHECK_FEATURE_ACTIVE (bmi, BMI1);
  fails += CHECK_FEATURE_ACTIVE (bmi2, BMI2);
#endif
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (cldemote, CLDEMOTE);
  fails += CHECK_FEATURE_ACTIVE (clflushopt, CLFLUSHOPT);
  fails += CHECK_FEATURE_ACTIVE (clwb, CLWB);
#endif
  fails += CHECK_FEATURE_ACTIVE (cmov, CMOV);
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (cmpxchg16b, CMPXCHG16B);
  fails += CHECK_FEATURE_ACTIVE (cmpxchg8b, CX8);
  fails += CHECK_FEATURE_ACTIVE (enqcmd, ENQCMD);
  fails += CHECK_FEATURE_ACTIVE (f16c, F16C);
#endif
#if __GNUC_PREREQ (4, 9)
  fails += CHECK_FEATURE_ACTIVE (fma, FMA);
  fails += CHECK_FEATURE_ACTIVE (fma4, FMA4);
#endif
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_PRESENT (fsgsbase, FSGSBASE);
  fails += CHECK_FEATURE_ACTIVE (fxsave, FXSR);
#endif
#if __GNUC_PREREQ (8, 0)
  fails += CHECK_FEATURE_ACTIVE (gfni, GFNI);
#endif
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (hle, HLE);
  fails += CHECK_FEATURE_PRESENT (ibt, IBT);
  fails += CHECK_FEATURE_ACTIVE (lahf_lm, LAHF64_SAHF64);
  fails += CHECK_FEATURE_PRESENT (lm, LM);
  fails += CHECK_FEATURE_ACTIVE (lwp, LWP);
  fails += CHECK_FEATURE_ACTIVE (lzcnt, LZCNT);
#endif
  fails += CHECK_FEATURE_ACTIVE (mmx, MMX);
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (movbe, MOVBE);
  fails += CHECK_FEATURE_ACTIVE (movdiri, MOVDIRI);
  fails += CHECK_FEATURE_ACTIVE (movdir64b, MOVDIR64B);
  fails += CHECK_FEATURE_ACTIVE (osxsave, OSXSAVE);
  fails += CHECK_FEATURE_ACTIVE (pconfig, PCONFIG);
  fails += CHECK_FEATURE_ACTIVE (pku, PKU);
#endif
  fails += CHECK_FEATURE_ACTIVE (popcnt, POPCNT);
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (prefetchwt1, PREFETCHWT1);
  fails += CHECK_FEATURE_ACTIVE (ptwrite, PTWRITE);
  fails += CHECK_FEATURE_ACTIVE (rdpid, RDPID);
  fails += CHECK_FEATURE_ACTIVE (rdrnd, RDRAND);
  fails += CHECK_FEATURE_ACTIVE (rdseed, RDSEED);
  fails += CHECK_FEATURE_PRESENT (rtm, RTM);
  fails += CHECK_FEATURE_ACTIVE (serialize, SERIALIZE);
  fails += CHECK_FEATURE_ACTIVE (sha, SHA);
  fails += CHECK_FEATURE_PRESENT (shstk, SHSTK);
#endif
  fails += CHECK_FEATURE_ACTIVE (sse, SSE);
  fails += CHECK_FEATURE_ACTIVE (sse2, SSE2);
  fails += CHECK_FEATURE_ACTIVE (sse3, SSE3);
  fails += CHECK_FEATURE_ACTIVE (sse4.1, SSE4_1);
  fails += CHECK_FEATURE_ACTIVE (sse4.2, SSE4_2);
#if __GNUC_PREREQ (4, 9)
  fails += CHECK_FEATURE_ACTIVE (sse4a, SSE4A);
#endif
  fails += CHECK_FEATURE_ACTIVE (ssse3, SSSE3);
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (tbm, TBM);
  fails += CHECK_FEATURE_ACTIVE (tsxldtrk, TSXLDTRK);
  fails += CHECK_FEATURE_ACTIVE (vaes, VAES);
#endif
#if __GNUC_PREREQ (8, 0)
  fails += CHECK_FEATURE_ACTIVE (vpclmulqdq, VPCLMULQDQ);
#endif
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (waitpkg, WAITPKG);
  fails += CHECK_FEATURE_ACTIVE (wbnoinvd, WBNOINVD);
#endif
#if __GNUC_PREREQ (4, 9)
  fails += CHECK_FEATURE_ACTIVE (xop, XOP);
#endif
#if __GNUC_PREREQ (11, 0)
  fails += CHECK_FEATURE_ACTIVE (xsave, XSAVE);
  fails += CHECK_FEATURE_ACTIVE (xsavec, XSAVEC);
  fails += CHECK_FEATURE_ACTIVE (xsaveopt, XSAVEOPT);
  fails += CHECK_FEATURE_PRESENT (xsaves, XSAVES);
#endif

  printf ("%d differences between __builtin_cpu_supports and glibc code.\n",
	  fails);

  return (fails != 0);
}

#include "../../../test-skeleton.c"
