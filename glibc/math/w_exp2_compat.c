/*
 * wrapper exp2(x)
 */

#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-double.h>

#if LIBM_SVID_COMPAT && (SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_29) \
			 || defined NO_LONG_DOUBLE \
			 || defined LONG_DOUBLE_COMPAT)
double
__exp2_compat (double x)
{
  double z = __ieee754_exp2 (x);
  if (__builtin_expect (!isfinite (z) || z == 0, 0)
      && isfinite (x) && _LIB_VERSION != _IEEE_)
    /* exp2 overflow: 44, exp2 underflow: 45 */
    return __kernel_standard (x, x, 44 + !!signbit (x));

  return z;
}
# if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_29)
compat_symbol (libm, __exp2_compat, exp2, GLIBC_2_1);
# endif
# ifdef NO_LONG_DOUBLE
weak_alias (__exp2_compat, exp2l)
# endif
# ifdef LONG_DOUBLE_COMPAT
/* Work around gas bug "multiple versions for symbol".  */
weak_alias (__exp2_compat, __exp2_compat_alias)

LONG_DOUBLE_COMPAT_CHOOSE_libm_exp2l (
  compat_symbol (libm, __exp2_compat_alias, exp2l, FIRST_VERSION_libm_exp2l), );
# endif
#endif
