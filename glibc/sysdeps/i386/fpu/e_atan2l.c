/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for `long double' by Ulrich Drepper <drepper@cygnus.com>.
 */

#include <math_private.h>
#include <libm-alias-finite.h>

long double
__ieee754_atan2l (long double y, long double x)
{
  long double res;

  asm ("fpatan" : "=t" (res) : "u" (y), "0" (x) : "st(1)");

  return res;
}
libm_alias_finite (__ieee754_atan2l, __atan2l)
