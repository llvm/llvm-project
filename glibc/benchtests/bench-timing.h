/* Define timing macros.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#undef attribute_hidden
#define attribute_hidden
#ifdef USE_CLOCK_GETTIME
# include <sysdeps/generic/hp-timing.h>
#else
# include <hp-timing.h>
#endif
#include <stdint.h>

#define GL(x) _##x
#define GLRO(x) _##x
typedef hp_timing_t timing_t;

#define TIMING_TYPE "hp_timing"

#define TIMING_NOW(var) HP_TIMING_NOW (var)
#define TIMING_DIFF(diff, start, end) HP_TIMING_DIFF ((diff), (start), (end))
#define TIMING_ACCUM(sum, diff) HP_TIMING_ACCUM_NT ((sum), (diff))

#define TIMING_PRINT_MEAN(d_total_s, d_iters) \
  printf ("\t%g", (d_total_s) / (d_iters))
