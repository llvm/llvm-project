/*
 * wrapper exp2f(x)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>

#if LIBM_SVID_COMPAT && SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_27)
float
__exp2f_compat (float x)
{
  float z = __ieee754_exp2f (x);
  if (__builtin_expect (!isfinite (z) || z == 0, 0)
      && isfinite (x) && _LIB_VERSION != _IEEE_)
    /* exp2 overflow: 144, exp2 underflow: 145 */
    return __kernel_standard_f (x, x, 144 + !!signbit (x));

  return z;
}
compat_symbol (libm, __exp2f_compat, exp2f, GLIBC_2_1);
#endif
