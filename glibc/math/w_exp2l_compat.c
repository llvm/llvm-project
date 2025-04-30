/*
 * wrapper exp2l(x)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>

#if LIBM_SVID_COMPAT
long double
__exp2l (long double x)
{
  long double z = __ieee754_exp2l (x);
  if (__builtin_expect (!isfinite (z) || z == 0, 0)
      && isfinite (x) && _LIB_VERSION != _IEEE_)
    /* exp2 overflow: 244, exp2 underflow: 245 */
    return __kernel_standard_l (x, x, 244 + !!signbit (x));

  return z;
}
libm_alias_ldouble (__exp2, exp2)
#endif
