#include <math.h>
#include <errno.h>
#include <limits.h>
#include <math_private.h>
#include <fenv.h>

#if _GL_HAS_BUILTIN_ILOGB
long int
M_DECL_FUNC (__llogb) (FLOAT x)
{
  int r;
  /* Check for exceptional cases.  */
  if (! M_SUF(__builtin_test_dc_ilogb) (x, 0x7f))
    r = M_SUF (__builtin_ilogb) (x);
  else
    /* Fallback to the generic ilogb if x is NaN, Inf or subnormal.  */
    r = M_SUF (__ieee754_ilogb) (x);
  long int lr = r;
  if (__glibc_unlikely (r == FP_ILOGB0)
      || __glibc_unlikely (r == FP_ILOGBNAN)
      || __glibc_unlikely (r == INT_MAX))
    {
#if LONG_MAX != INT_MAX
      if (r == FP_ILOGB0)
	lr = FP_LLOGB0;
      else if (r == FP_ILOGBNAN)
	lr = FP_LLOGBNAN;
      else
	lr = LONG_MAX;
#endif
      __set_errno (EDOM);
      __feraiseexcept (FE_INVALID);
    }
  return lr;
}
declare_mgen_alias (__llogb, llogb)
#else
#include <math/w_llogb_template.c>
#endif
