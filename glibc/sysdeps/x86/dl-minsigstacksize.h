/* Emulate AT_MINSIGSTKSZ.  Generic x86 version.
   Copyright (C) 2020 Free Software Foundation, Inc.

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

/* Emulate AT_MINSIGSTKSZ with XSAVE. */

static inline void
dl_check_minsigstacksize (const struct cpu_features *cpu_features)
{
  /* NB: Default to a constant MINSIGSTKSZ.  */
  _Static_assert (__builtin_constant_p (MINSIGSTKSZ),
		  "MINSIGSTKSZ is constant");
  GLRO(dl_minsigstacksize) = MINSIGSTKSZ;
}
