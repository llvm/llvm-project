/* Common definition for libmathvec ifunc selections optimized with
   SSE4.1.
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

#undef PASTER2
#define PASTER2(x,y)	x##_##y

extern void REDIRECT_NAME (void);
extern __typeof (REDIRECT_NAME) OPTIMIZE (sse2) attribute_hidden;
extern __typeof (REDIRECT_NAME) OPTIMIZE (sse4) attribute_hidden;

static inline void *
IFUNC_SELECTOR (void)
{
  const struct cpu_features* cpu_features = __get_cpu_features ();

  if (CPU_FEATURE_USABLE_P (cpu_features, SSE4_1))
    return OPTIMIZE (sse4);

  return OPTIMIZE (sse2);
}
