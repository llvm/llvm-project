/* Enumerate available IFUNC implementations of a function.  x86-64 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <assert.h>
#include <string.h>
#include <wchar.h>
#include <ifunc-impl-list.h>
#include <sysdep.h>
#include "init-arch.h"

/* Maximum number of IFUNC implementations.  */
#define MAX_IFUNC	5

/* Fill ARRAY of MAX elements with IFUNC implementations for function
   NAME supported on target machine and return the number of valid
   entries.  */

size_t
__libc_ifunc_impl_list (const char *name, struct libc_ifunc_impl *array,
			size_t max)
{
  assert (max >= MAX_IFUNC);

  size_t i = 0;

  /* Support sysdeps/x86_64/multiarch/memchr.c.  */
  IFUNC_IMPL (i, name, memchr,
	      IFUNC_IMPL_ADD (array, i, memchr,
			      CPU_FEATURE_USABLE (AVX2),
			      __memchr_avx2)
	      IFUNC_IMPL_ADD (array, i, memchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, memchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memchr_evex)
	      IFUNC_IMPL_ADD (array, i, memchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memchr_evex_rtm)
	      IFUNC_IMPL_ADD (array, i, memchr, 1, __memchr_sse2))

  /* Support sysdeps/x86_64/multiarch/memcmp.c.  */
  IFUNC_IMPL (i, name, memcmp,
	      IFUNC_IMPL_ADD (array, i, memcmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (MOVBE)),
			      __memcmp_avx2_movbe)
	      IFUNC_IMPL_ADD (array, i, memcmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (MOVBE)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memcmp_avx2_movbe_rtm)
	      IFUNC_IMPL_ADD (array, i, memcmp,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (MOVBE)),
			      __memcmp_evex_movbe)
	      IFUNC_IMPL_ADD (array, i, memcmp, CPU_FEATURE_USABLE (SSE4_1),
			      __memcmp_sse4_1)
	      IFUNC_IMPL_ADD (array, i, memcmp, CPU_FEATURE_USABLE (SSSE3),
			      __memcmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, memcmp, 1, __memcmp_sse2))

#ifdef SHARED
  /* Support sysdeps/x86_64/multiarch/memmove_chk.c.  */
  IFUNC_IMPL (i, name, __memmove_chk,
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX512F),
			      __memmove_chk_avx512_no_vzeroupper)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_chk_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_chk_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX),
			      __memmove_chk_avx_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX),
			      __memmove_chk_avx_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memmove_chk_avx_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memmove_chk_avx_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_chk_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_chk_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memmove_chk_ssse3_back)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memmove_chk_ssse3)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk, 1,
			      __memmove_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk, 1,
			      __memmove_chk_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk, 1,
			      __memmove_chk_erms))
#endif

  /* Support sysdeps/x86_64/multiarch/memmove.c.  */
  IFUNC_IMPL (i, name, memmove,
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX),
			      __memmove_avx_unaligned)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX),
			      __memmove_avx_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memmove_avx_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memmove_avx_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX512F),
			      __memmove_avx512_no_vzeroupper)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, memmove,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memmove_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memmove, CPU_FEATURE_USABLE (SSSE3),
			      __memmove_ssse3_back)
	      IFUNC_IMPL_ADD (array, i, memmove, CPU_FEATURE_USABLE (SSSE3),
			      __memmove_ssse3)
	      IFUNC_IMPL_ADD (array, i, memmove, 1, __memmove_erms)
	      IFUNC_IMPL_ADD (array, i, memmove, 1,
			      __memmove_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, memmove, 1,
			      __memmove_sse2_unaligned_erms))

  /* Support sysdeps/x86_64/multiarch/memrchr.c.  */
  IFUNC_IMPL (i, name, memrchr,
	      IFUNC_IMPL_ADD (array, i, memrchr,
			      CPU_FEATURE_USABLE (AVX2),
			      __memrchr_avx2)
	      IFUNC_IMPL_ADD (array, i, memrchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memrchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, memrchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __memrchr_evex)

	      IFUNC_IMPL_ADD (array, i, memrchr, 1, __memrchr_sse2))

#ifdef SHARED
  /* Support sysdeps/x86_64/multiarch/memset_chk.c.  */
  IFUNC_IMPL (i, name, __memset_chk,
	      IFUNC_IMPL_ADD (array, i, __memset_chk, 1,
			      __memset_chk_erms)
	      IFUNC_IMPL_ADD (array, i, __memset_chk, 1,
			      __memset_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memset_chk, 1,
			      __memset_chk_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      CPU_FEATURE_USABLE (AVX2),
			      __memset_chk_avx2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      CPU_FEATURE_USABLE (AVX2),
			      __memset_chk_avx2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memset_chk_avx2_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memset_chk_avx2_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_chk_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_chk_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_chk_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_chk_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      CPU_FEATURE_USABLE (AVX512F),
			      __memset_chk_avx512_no_vzeroupper)
	      )
#endif

  /* Support sysdeps/x86_64/multiarch/memset.c.  */
  IFUNC_IMPL (i, name, memset,
	      IFUNC_IMPL_ADD (array, i, memset, 1,
			      __memset_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, memset, 1,
			      __memset_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memset, 1, __memset_erms)
	      IFUNC_IMPL_ADD (array, i, memset,
			      CPU_FEATURE_USABLE (AVX2),
			      __memset_avx2_unaligned)
	      IFUNC_IMPL_ADD (array, i, memset,
			      CPU_FEATURE_USABLE (AVX2),
			      __memset_avx2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memset,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memset_avx2_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, memset,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memset_avx2_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, memset,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, memset,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memset,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memset,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __memset_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, memset,
			      CPU_FEATURE_USABLE (AVX512F),
			      __memset_avx512_no_vzeroupper)
	     )

  /* Support sysdeps/x86_64/multiarch/rawmemchr.c.  */
  IFUNC_IMPL (i, name, rawmemchr,
	      IFUNC_IMPL_ADD (array, i, rawmemchr,
			      CPU_FEATURE_USABLE (AVX2),
			      __rawmemchr_avx2)
	      IFUNC_IMPL_ADD (array, i, rawmemchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __rawmemchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, rawmemchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __rawmemchr_evex)
	      IFUNC_IMPL_ADD (array, i, rawmemchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __rawmemchr_evex_rtm)
	      IFUNC_IMPL_ADD (array, i, rawmemchr, 1, __rawmemchr_sse2))

  /* Support sysdeps/x86_64/multiarch/strlen.c.  */
  IFUNC_IMPL (i, name, strlen,
	      IFUNC_IMPL_ADD (array, i, strlen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strlen_avx2)
	      IFUNC_IMPL_ADD (array, i, strlen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strlen_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strlen,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strlen_evex)
	      IFUNC_IMPL_ADD (array, i, strlen, 1, __strlen_sse2))

  /* Support sysdeps/x86_64/multiarch/strnlen.c.  */
  IFUNC_IMPL (i, name, strnlen,
	      IFUNC_IMPL_ADD (array, i, strnlen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strnlen_avx2)
	      IFUNC_IMPL_ADD (array, i, strnlen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strnlen_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strnlen,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strnlen_evex)
	      IFUNC_IMPL_ADD (array, i, strnlen, 1, __strnlen_sse2))

  /* Support sysdeps/x86_64/multiarch/stpncpy.c.  */
  IFUNC_IMPL (i, name, stpncpy,
	      IFUNC_IMPL_ADD (array, i, stpncpy, CPU_FEATURE_USABLE (SSSE3),
			      __stpncpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, stpncpy, CPU_FEATURE_USABLE (AVX2),
			      __stpncpy_avx2)
	      IFUNC_IMPL_ADD (array, i, stpncpy,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __stpncpy_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, stpncpy,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __stpncpy_evex)
	      IFUNC_IMPL_ADD (array, i, stpncpy, 1,
			      __stpncpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, stpncpy, 1, __stpncpy_sse2))

  /* Support sysdeps/x86_64/multiarch/stpcpy.c.  */
  IFUNC_IMPL (i, name, stpcpy,
	      IFUNC_IMPL_ADD (array, i, stpcpy, CPU_FEATURE_USABLE (SSSE3),
			      __stpcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, stpcpy, CPU_FEATURE_USABLE (AVX2),
			      __stpcpy_avx2)
	      IFUNC_IMPL_ADD (array, i, stpcpy,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __stpcpy_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, stpcpy,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __stpcpy_evex)
	      IFUNC_IMPL_ADD (array, i, stpcpy, 1, __stpcpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, stpcpy, 1, __stpcpy_sse2))

  /* Support sysdeps/x86_64/multiarch/strcasecmp_l.c.  */
  IFUNC_IMPL (i, name, strcasecmp,
	      IFUNC_IMPL_ADD (array, i, strcasecmp,
			      CPU_FEATURE_USABLE (AVX),
			      __strcasecmp_avx)
	      IFUNC_IMPL_ADD (array, i, strcasecmp,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strcasecmp_sse42)
	      IFUNC_IMPL_ADD (array, i, strcasecmp,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strcasecmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcasecmp, 1, __strcasecmp_sse2))

  /* Support sysdeps/x86_64/multiarch/strcasecmp_l.c.  */
  IFUNC_IMPL (i, name, strcasecmp_l,
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l,
			      CPU_FEATURE_USABLE (AVX),
			      __strcasecmp_l_avx)
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strcasecmp_l_sse42)
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strcasecmp_l_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l, 1,
			      __strcasecmp_l_sse2))

  /* Support sysdeps/x86_64/multiarch/strcat.c.  */
  IFUNC_IMPL (i, name, strcat,
	      IFUNC_IMPL_ADD (array, i, strcat, CPU_FEATURE_USABLE (AVX2),
			      __strcat_avx2)
	      IFUNC_IMPL_ADD (array, i, strcat,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strcat_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strcat,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __strcat_evex)
	      IFUNC_IMPL_ADD (array, i, strcat, CPU_FEATURE_USABLE (SSSE3),
			      __strcat_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcat, 1, __strcat_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, strcat, 1, __strcat_sse2))

  /* Support sysdeps/x86_64/multiarch/strchr.c.  */
  IFUNC_IMPL (i, name, strchr,
	      IFUNC_IMPL_ADD (array, i, strchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strchr_avx2)
	      IFUNC_IMPL_ADD (array, i, strchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strchr_evex)
	      IFUNC_IMPL_ADD (array, i, strchr, 1, __strchr_sse2_no_bsf)
	      IFUNC_IMPL_ADD (array, i, strchr, 1, __strchr_sse2))

  /* Support sysdeps/x86_64/multiarch/strchrnul.c.  */
  IFUNC_IMPL (i, name, strchrnul,
	      IFUNC_IMPL_ADD (array, i, strchrnul,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strchrnul_avx2)
	      IFUNC_IMPL_ADD (array, i, strchrnul,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strchrnul_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strchrnul,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strchrnul_evex)
	      IFUNC_IMPL_ADD (array, i, strchrnul, 1, __strchrnul_sse2))

  /* Support sysdeps/x86_64/multiarch/strrchr.c.  */
  IFUNC_IMPL (i, name, strrchr,
	      IFUNC_IMPL_ADD (array, i, strrchr,
			      CPU_FEATURE_USABLE (AVX2),
			      __strrchr_avx2)
	      IFUNC_IMPL_ADD (array, i, strrchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strrchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strrchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __strrchr_evex)
	      IFUNC_IMPL_ADD (array, i, strrchr, 1, __strrchr_sse2))

  /* Support sysdeps/x86_64/multiarch/strcmp.c.  */
  IFUNC_IMPL (i, name, strcmp,
	      IFUNC_IMPL_ADD (array, i, strcmp,
			      CPU_FEATURE_USABLE (AVX2),
			      __strcmp_avx2)
	      IFUNC_IMPL_ADD (array, i, strcmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strcmp_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strcmp,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __strcmp_evex)
	      IFUNC_IMPL_ADD (array, i, strcmp, CPU_FEATURE_USABLE (SSE4_2),
			      __strcmp_sse42)
	      IFUNC_IMPL_ADD (array, i, strcmp, CPU_FEATURE_USABLE (SSSE3),
			      __strcmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcmp, 1, __strcmp_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, strcmp, 1, __strcmp_sse2))

  /* Support sysdeps/x86_64/multiarch/strcpy.c.  */
  IFUNC_IMPL (i, name, strcpy,
	      IFUNC_IMPL_ADD (array, i, strcpy, CPU_FEATURE_USABLE (AVX2),
			      __strcpy_avx2)
	      IFUNC_IMPL_ADD (array, i, strcpy,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strcpy_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strcpy,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __strcpy_evex)
	      IFUNC_IMPL_ADD (array, i, strcpy, CPU_FEATURE_USABLE (SSSE3),
			      __strcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcpy, 1, __strcpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, strcpy, 1, __strcpy_sse2))

  /* Support sysdeps/x86_64/multiarch/strcspn.c.  */
  IFUNC_IMPL (i, name, strcspn,
	      IFUNC_IMPL_ADD (array, i, strcspn, CPU_FEATURE_USABLE (SSE4_2),
			      __strcspn_sse42)
	      IFUNC_IMPL_ADD (array, i, strcspn, 1, __strcspn_sse2))

  /* Support sysdeps/x86_64/multiarch/strncase_l.c.  */
  IFUNC_IMPL (i, name, strncasecmp,
	      IFUNC_IMPL_ADD (array, i, strncasecmp,
			      CPU_FEATURE_USABLE (AVX),
			      __strncasecmp_avx)
	      IFUNC_IMPL_ADD (array, i, strncasecmp,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strncasecmp_sse42)
	      IFUNC_IMPL_ADD (array, i, strncasecmp,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strncasecmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncasecmp, 1,
			      __strncasecmp_sse2))

  /* Support sysdeps/x86_64/multiarch/strncase_l.c.  */
  IFUNC_IMPL (i, name, strncasecmp_l,
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l,
			      CPU_FEATURE_USABLE (AVX),
			      __strncasecmp_l_avx)
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strncasecmp_l_sse42)
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strncasecmp_l_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l, 1,
			      __strncasecmp_l_sse2))

  /* Support sysdeps/x86_64/multiarch/strncat.c.  */
  IFUNC_IMPL (i, name, strncat,
	      IFUNC_IMPL_ADD (array, i, strncat, CPU_FEATURE_USABLE (AVX2),
			      __strncat_avx2)
	      IFUNC_IMPL_ADD (array, i, strncat,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strncat_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strncat,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __strncat_evex)
	      IFUNC_IMPL_ADD (array, i, strncat, CPU_FEATURE_USABLE (SSSE3),
			      __strncat_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncat, 1,
			      __strncat_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, strncat, 1, __strncat_sse2))

  /* Support sysdeps/x86_64/multiarch/strncpy.c.  */
  IFUNC_IMPL (i, name, strncpy,
	      IFUNC_IMPL_ADD (array, i, strncpy, CPU_FEATURE_USABLE (AVX2),
			      __strncpy_avx2)
	      IFUNC_IMPL_ADD (array, i, strncpy,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strncpy_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strncpy,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __strncpy_evex)
	      IFUNC_IMPL_ADD (array, i, strncpy, CPU_FEATURE_USABLE (SSSE3),
			      __strncpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncpy, 1,
			      __strncpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, strncpy, 1, __strncpy_sse2))

  /* Support sysdeps/x86_64/multiarch/strpbrk.c.  */
  IFUNC_IMPL (i, name, strpbrk,
	      IFUNC_IMPL_ADD (array, i, strpbrk, CPU_FEATURE_USABLE (SSE4_2),
			      __strpbrk_sse42)
	      IFUNC_IMPL_ADD (array, i, strpbrk, 1, __strpbrk_sse2))


  /* Support sysdeps/x86_64/multiarch/strspn.c.  */
  IFUNC_IMPL (i, name, strspn,
	      IFUNC_IMPL_ADD (array, i, strspn, CPU_FEATURE_USABLE (SSE4_2),
			      __strspn_sse42)
	      IFUNC_IMPL_ADD (array, i, strspn, 1, __strspn_sse2))

  /* Support sysdeps/x86_64/multiarch/strstr.c.  */
  IFUNC_IMPL (i, name, strstr,
	      IFUNC_IMPL_ADD (array, i, strstr, 1, __strstr_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, strstr, 1, __strstr_sse2))

  /* Support sysdeps/x86_64/multiarch/wcschr.c.  */
  IFUNC_IMPL (i, name, wcschr,
	      IFUNC_IMPL_ADD (array, i, wcschr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcschr_avx2)
	      IFUNC_IMPL_ADD (array, i, wcschr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wcschr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wcschr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcschr_evex)
	      IFUNC_IMPL_ADD (array, i, wcschr, 1, __wcschr_sse2))

  /* Support sysdeps/x86_64/multiarch/wcsrchr.c.  */
  IFUNC_IMPL (i, name, wcsrchr,
	      IFUNC_IMPL_ADD (array, i, wcsrchr,
			      CPU_FEATURE_USABLE (AVX2),
			      __wcsrchr_avx2)
	      IFUNC_IMPL_ADD (array, i, wcsrchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wcsrchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wcsrchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcsrchr_evex)
	      IFUNC_IMPL_ADD (array, i, wcsrchr, 1, __wcsrchr_sse2))

  /* Support sysdeps/x86_64/multiarch/wcscmp.c.  */
  IFUNC_IMPL (i, name, wcscmp,
	      IFUNC_IMPL_ADD (array, i, wcscmp,
			      CPU_FEATURE_USABLE (AVX2),
			      __wcscmp_avx2)
	      IFUNC_IMPL_ADD (array, i, wcscmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wcscmp_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wcscmp,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcscmp_evex)
	      IFUNC_IMPL_ADD (array, i, wcscmp, 1, __wcscmp_sse2))

  /* Support sysdeps/x86_64/multiarch/wcsncmp.c.  */
  IFUNC_IMPL (i, name, wcsncmp,
	      IFUNC_IMPL_ADD (array, i, wcsncmp,
			      CPU_FEATURE_USABLE (AVX2),
			      __wcsncmp_avx2)
	      IFUNC_IMPL_ADD (array, i, wcsncmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wcsncmp_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wcsncmp,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcsncmp_evex)
	      IFUNC_IMPL_ADD (array, i, wcsncmp, 1, __wcsncmp_sse2))

  /* Support sysdeps/x86_64/multiarch/wcscpy.c.  */
  IFUNC_IMPL (i, name, wcscpy,
	      IFUNC_IMPL_ADD (array, i, wcscpy, CPU_FEATURE_USABLE (SSSE3),
			      __wcscpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, wcscpy, 1, __wcscpy_sse2))

  /* Support sysdeps/x86_64/multiarch/wcslen.c.  */
  IFUNC_IMPL (i, name, wcslen,
	      IFUNC_IMPL_ADD (array, i, wcslen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcslen_avx2)
	      IFUNC_IMPL_ADD (array, i, wcslen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wcslen_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wcslen,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcslen_evex)
	      IFUNC_IMPL_ADD (array, i, wcslen,
			      CPU_FEATURE_USABLE (SSE4_1),
			      __wcslen_sse4_1)
	      IFUNC_IMPL_ADD (array, i, wcslen, 1, __wcslen_sse2))

  /* Support sysdeps/x86_64/multiarch/wcsnlen.c.  */
  IFUNC_IMPL (i, name, wcsnlen,
	      IFUNC_IMPL_ADD (array, i, wcsnlen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcsnlen_avx2)
	      IFUNC_IMPL_ADD (array, i, wcsnlen,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wcsnlen_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wcsnlen,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wcsnlen_evex)
	      IFUNC_IMPL_ADD (array, i, wcsnlen,
			      CPU_FEATURE_USABLE (SSE4_1),
			      __wcsnlen_sse4_1)
	      IFUNC_IMPL_ADD (array, i, wcsnlen, 1, __wcsnlen_sse2))

  /* Support sysdeps/x86_64/multiarch/wmemchr.c.  */
  IFUNC_IMPL (i, name, wmemchr,
	      IFUNC_IMPL_ADD (array, i, wmemchr,
			      CPU_FEATURE_USABLE (AVX2),
			      __wmemchr_avx2)
	      IFUNC_IMPL_ADD (array, i, wmemchr,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wmemchr_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, wmemchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wmemchr_evex)
	      IFUNC_IMPL_ADD (array, i, wmemchr,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wmemchr_evex_rtm)
	      IFUNC_IMPL_ADD (array, i, wmemchr, 1, __wmemchr_sse2))

  /* Support sysdeps/x86_64/multiarch/wmemcmp.c.  */
  IFUNC_IMPL (i, name, wmemcmp,
	      IFUNC_IMPL_ADD (array, i, wmemcmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (MOVBE)),
			      __wmemcmp_avx2_movbe)
	      IFUNC_IMPL_ADD (array, i, wmemcmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (MOVBE)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wmemcmp_avx2_movbe_rtm)
	      IFUNC_IMPL_ADD (array, i, wmemcmp,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)
			       && CPU_FEATURE_USABLE (MOVBE)),
			      __wmemcmp_evex_movbe)
	      IFUNC_IMPL_ADD (array, i, wmemcmp, CPU_FEATURE_USABLE (SSE4_1),
			      __wmemcmp_sse4_1)
	      IFUNC_IMPL_ADD (array, i, wmemcmp, CPU_FEATURE_USABLE (SSSE3),
			      __wmemcmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, wmemcmp, 1, __wmemcmp_sse2))

  /* Support sysdeps/x86_64/multiarch/wmemset.c.  */
  IFUNC_IMPL (i, name, wmemset,
	      IFUNC_IMPL_ADD (array, i, wmemset, 1,
			      __wmemset_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, wmemset,
			      CPU_FEATURE_USABLE (AVX2),
			      __wmemset_avx2_unaligned)
	      IFUNC_IMPL_ADD (array, i, wmemset,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __wmemset_avx2_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, wmemset,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wmemset_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, wmemset,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wmemset_avx512_unaligned))

#ifdef SHARED
  /* Support sysdeps/x86_64/multiarch/memcpy_chk.c.  */
  IFUNC_IMPL (i, name, __memcpy_chk,
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX512F),
			      __memcpy_chk_avx512_no_vzeroupper)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_chk_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_chk_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX),
			      __memcpy_chk_avx_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX),
			      __memcpy_chk_avx_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memcpy_chk_avx_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memcpy_chk_avx_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_chk_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_chk_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_chk_ssse3_back)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_chk_ssse3)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk, 1,
			      __memcpy_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk, 1,
			      __memcpy_chk_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk, 1,
			      __memcpy_chk_erms))
#endif

  /* Support sysdeps/x86_64/multiarch/memcpy.c.  */
  IFUNC_IMPL (i, name, memcpy,
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX),
			      __memcpy_avx_unaligned)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX),
			      __memcpy_avx_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memcpy_avx_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __memcpy_avx_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memcpy, CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_ssse3_back)
	      IFUNC_IMPL_ADD (array, i, memcpy, CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX512F),
			      __memcpy_avx512_no_vzeroupper)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __memcpy_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, __memcpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, memcpy, 1,
			      __memcpy_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, __memcpy_erms))

#ifdef SHARED
  /* Support sysdeps/x86_64/multiarch/mempcpy_chk.c.  */
  IFUNC_IMPL (i, name, __mempcpy_chk,
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX512F),
			      __mempcpy_chk_avx512_no_vzeroupper)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_chk_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_chk_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX),
			      __mempcpy_chk_avx_unaligned)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX),
			      __mempcpy_chk_avx_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __mempcpy_chk_avx_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __mempcpy_chk_avx_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_chk_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_chk_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_chk_ssse3_back)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_chk_ssse3)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk, 1,
			      __mempcpy_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk, 1,
			      __mempcpy_chk_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk, 1,
			      __mempcpy_chk_erms))
#endif

  /* Support sysdeps/x86_64/multiarch/mempcpy.c.  */
  IFUNC_IMPL (i, name, mempcpy,
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX512F),
			      __mempcpy_avx512_no_vzeroupper)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_avx512_unaligned)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_avx512_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX),
			      __mempcpy_avx_unaligned)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX),
			      __mempcpy_avx_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __mempcpy_avx_unaligned_rtm)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      (CPU_FEATURE_USABLE (AVX)
			       && CPU_FEATURE_USABLE (RTM)),
			      __mempcpy_avx_unaligned_erms_rtm)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      CPU_FEATURE_USABLE (AVX512VL),
			      __mempcpy_evex_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, mempcpy, CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_ssse3_back)
	      IFUNC_IMPL_ADD (array, i, mempcpy, CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1,
			      __mempcpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1,
			      __mempcpy_sse2_unaligned_erms)
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1, __mempcpy_erms))

  /* Support sysdeps/x86_64/multiarch/strncmp.c.  */
  IFUNC_IMPL (i, name, strncmp,
	      IFUNC_IMPL_ADD (array, i, strncmp,
			      CPU_FEATURE_USABLE (AVX2),
			      __strncmp_avx2)
	      IFUNC_IMPL_ADD (array, i, strncmp,
			      (CPU_FEATURE_USABLE (AVX2)
			       && CPU_FEATURE_USABLE (RTM)),
			      __strncmp_avx2_rtm)
	      IFUNC_IMPL_ADD (array, i, strncmp,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)),
			      __strncmp_evex)
	      IFUNC_IMPL_ADD (array, i, strncmp, CPU_FEATURE_USABLE (SSE4_2),
			      __strncmp_sse42)
	      IFUNC_IMPL_ADD (array, i, strncmp, CPU_FEATURE_USABLE (SSSE3),
			      __strncmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncmp, 1, __strncmp_sse2))

#ifdef SHARED
  /* Support sysdeps/x86_64/multiarch/wmemset_chk.c.  */
  IFUNC_IMPL (i, name, __wmemset_chk,
	      IFUNC_IMPL_ADD (array, i, __wmemset_chk, 1,
			      __wmemset_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __wmemset_chk,
			      CPU_FEATURE_USABLE (AVX2),
			      __wmemset_chk_avx2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __wmemset_chk,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wmemset_chk_evex_unaligned)
	      IFUNC_IMPL_ADD (array, i, __wmemset_chk,
			      (CPU_FEATURE_USABLE (AVX512VL)
			       && CPU_FEATURE_USABLE (AVX512BW)
			       && CPU_FEATURE_USABLE (BMI2)),
			      __wmemset_chk_avx512_unaligned))
#endif

  return i;
}
