/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 *
 * Adapted for `long double' by Ulrich Drepper <drepper@cygnus.com>.
 */

#include <math_private.h>
#include <libm-alias-finite.h>

long double
__ieee754_fmodl (long double x, long double y)
{
  long double res;

  asm ("1:\tfprem\n"
       "fstsw   %%ax\n"
       "sahf\n"
       "jp      1b\n"
       "fstp    %%st(1)"
       : "=t" (res) : "0" (x), "u" (y) : "ax", "st(1)");
  return res;
}
libm_alias_finite (__ieee754_fmodl, __fmodl)
