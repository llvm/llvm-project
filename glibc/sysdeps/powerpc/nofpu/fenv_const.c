/* Constants for fenv_bits.h (soft float edition).
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   Contributed by Aldy Hernandez <aldyh@redhat.com>, 2002.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* We want to specify the bit pattern of the __fe_*_env constants, so
   pretend they're really `long long' instead of `double'.  */

/* If the default argument is used we use this value.  Disable all
   signalling exceptions as default.  */
const unsigned long long __fe_dfl_env __attribute__ ((aligned (8))) =
0x000000003e000000ULL;

/* The same representation is used for femode_t.  */
extern const unsigned long long __fe_dfl_mode
  __attribute__ ((aligned (8), alias ("__fe_dfl_env")));

/* Floating-point environment where none of the exceptions are masked.  */
const unsigned long long __fe_enabled_env __attribute__ ((aligned (8))) =
0x0000000000000000ULL;

/* Floating-point environment with the NI bit set.  No difference for
   soft float from the default environment.  */
strong_alias (__fe_dfl_env, __fe_nonieee_env)
