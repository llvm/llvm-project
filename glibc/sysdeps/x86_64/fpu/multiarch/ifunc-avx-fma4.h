/* Common definition for ifunc selections optimized with AVX, AVX2/FMA
   and FMA4.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <init-arch.h>

extern __typeof (REDIRECT_NAME) OPTIMIZE (sse2) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (avx) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (fma) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (fma4) attribute_hidden;

static inline void *
IFUNC_SELECTOR (void)
{
  const struct cpu_features* cpu_features = __get_cpu_features ();

  if (CPU_FEATURE_USABLE_P (cpu_features, FMA)
      && CPU_FEATURE_USABLE_P (cpu_features, AVX2))
    return OPTIMIZE (fma);

  if (CPU_FEATURE_USABLE_P (cpu_features, FMA4))
    return OPTIMIZE (fma4);

  if (CPU_FEATURE_USABLE_P (cpu_features, AVX))
    return OPTIMIZE (avx);

  return OPTIMIZE (sse2);
}
