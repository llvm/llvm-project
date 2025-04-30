/* Return classification value corresponding to argument.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997 and
		  Jakub Jelinek <jj@ultra.linux.cz>, 1999.

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

#include <math.h>

#include <math_private.h>
#include <math_ldbl_opt.h>

  /*
   *		hx                  lx
   * +NaN	7ffn nnnn nnnn nnnn xxxx xxxx xxxx xxxx
   * -NaN	fffn nnnn nnnn nnnn xxxx xxxx xxxx xxxx
   * +Inf	7ff0 0000 0000 0000 xxxx xxxx xxxx xxxx
   * -Inf	fff0 0000 0000 0000 xxxx xxxx xxxx xxxx
   * +0		0000 0000 0000 0000 xxxx xxxx xxxx xxxx
   * -0		8000 0000 0000 0000 xxxx xxxx xxxx xxxx
   * +normal	0360 0000 0000 0000 0000 0000 0000 0000 (smallest)
   * -normal	8360 0000 0000 0000 0000 0000 0000 0000 (smallest)
   * +normal	7fef ffff ffff ffff 7c8f ffff ffff fffe (largest)
   * +normal	ffef ffff ffff ffff fc8f ffff ffff fffe (largest)
   * +denorm	0360 0000 0000 0000 8000 0000 0000 0001 (largest)
   * -denorm	8360 0000 0000 0000 0000 0000 0000 0001 (largest)
   * +denorm	000n nnnn nnnn nnnn xxxx xxxx xxxx xxxx
   * -denorm	800n nnnn nnnn nnnn xxxx xxxx xxxx xxxx
   */

int
___fpclassifyl (long double x)
{
  uint64_t hx, lx;
  int retval = FP_NORMAL;
  double xhi, xlo;

  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  if ((hx & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) {
      /* +/-NaN or +/-Inf */
      if (hx & 0x000fffffffffffffULL) {
	  /* +/-NaN */
	  retval = FP_NAN;
      } else {
	  retval = FP_INFINITE;
      }
  } else {
      /* +/-zero or +/- normal or +/- denormal */
      if (hx & 0x7fffffffffffffffULL) {
	  /* +/- normal or +/- denormal */
	  if ((hx & 0x7ff0000000000000ULL) > 0x0360000000000000ULL) {
	      /* +/- normal */
	      retval = FP_NORMAL;
	  } else {
	      if ((hx & 0x7ff0000000000000ULL) == 0x0360000000000000ULL) {
		  EXTRACT_WORDS64 (lx, xlo);
		  if ((lx & 0x7fffffffffffffff)	/* lower is non-zero */
		  && ((lx^hx) & 0x8000000000000000ULL)) { /* and sign differs */
		      /* +/- denormal */
		      retval = FP_SUBNORMAL;
		  } else {
		      /* +/- normal */
		      retval = FP_NORMAL;
		  }
	      } else {
		  /* +/- denormal */
		  retval = FP_SUBNORMAL;
	      }
	  }
      } else {
	  /* +/- zero */
	  retval = FP_ZERO;
      }
  }

  return retval;
}
long_double_symbol (libm, ___fpclassifyl, __fpclassifyl);
#ifdef __LONG_DOUBLE_MATH_OPTIONAL
libm_hidden_ver (___fpclassifyl, __fpclassifyl)
#else
libm_hidden_def (__fpclassifyl)
#endif
