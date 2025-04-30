/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for `long double' by Ulrich Drepper <drepper@cygnus.com>.
 */

#include <math_private.h>
#include <libm-alias-finite.h>

long double
__ieee754_acosl (long double x)
{
  long double res;

  /* acosl = atanl (sqrtl((1-x) (1+x)) / x) */
  asm (	"fld	%%st\n"
	"fld1\n"
	"fsubp\n"
	"fld1\n"
	"fadd	%%st(2)\n"
	"fmulp\n"			/* 1 - x^2 */
	"fsqrt\n"			/* sqrtl (1 - x^2) */
	"fabs\n"
	"fxch	%%st(1)\n"
	"fpatan"
	: "=t" (res) : "0" (x) : "st(1)");
  return res;
}
libm_alias_finite (__ieee754_acosl, __acosl)
