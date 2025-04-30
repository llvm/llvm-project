/* Multiple versions of strchr.
   All versions must be listed in ifunc-impl-list.c.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

/* Define multiple versions only for the definition in libc.  */
#if IS_IN (libc)
# define strchr __redirect_strchr
# include <string.h>
# undef strchr

# define SYMBOL_NAME strchr
# include <init-arch.h>

extern __typeof (REDIRECT_NAME) OPTIMIZE (sse2) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (sse2_no_bsf) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (avx2) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (avx2_rtm) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (evex) attribute_hidden;

static inline void *
IFUNC_SELECTOR (void)
{
  const struct cpu_features* cpu_features = __get_cpu_features ();

  if (CPU_FEATURE_USABLE_P (cpu_features, AVX2)
      && CPU_FEATURE_USABLE_P (cpu_features, BMI2)
      && CPU_FEATURES_ARCH_P (cpu_features, AVX_Fast_Unaligned_Load))
    {
      if (CPU_FEATURE_USABLE_P (cpu_features, AVX512VL)
	  && CPU_FEATURE_USABLE_P (cpu_features, AVX512BW))
	return OPTIMIZE (evex);

      if (CPU_FEATURE_USABLE_P (cpu_features, RTM))
	return OPTIMIZE (avx2_rtm);

      if (!CPU_FEATURES_ARCH_P (cpu_features, Prefer_No_VZEROUPPER))
	return OPTIMIZE (avx2);
    }

  if (CPU_FEATURES_ARCH_P (cpu_features, Slow_BSF))
    return OPTIMIZE (sse2_no_bsf);

  return OPTIMIZE (sse2);
}

libc_ifunc_redirected (__redirect_strchr, strchr, IFUNC_SELECTOR ());
weak_alias (strchr, index)
# ifdef SHARED
__hidden_ver1 (strchr, __GI_strchr, __redirect_strchr)
  __attribute__((visibility ("hidden"))) __attribute_copy__ (strchr);
# endif
#endif
