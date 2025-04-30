/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#include <gmp.h>
#include "gmp-impl.h"

/* Convert a `double' to a multi-precision integer representing the
   significand scaled up by the highest possible number of significant bits
   of fraction (DBL_MANT_DIG), and an integral power of two (MPN frexp). */

mp_size_t
__mpn_extract_double (mp_ptr res_ptr, mp_size_t size,
		      int *expt, int *is_neg,
		      double value)
{
#error "__mpn_extract_double is not implemented for this floating point format"
}
