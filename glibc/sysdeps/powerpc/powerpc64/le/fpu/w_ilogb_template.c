#include <math.h>
#include <errno.h>
#include <limits.h>
#include <math_private.h>
#include <fenv.h>

#if _GL_HAS_BUILTIN_ILOGB
int
M_DECL_FUNC (__ilogb) (FLOAT x)
{
  int r;
  /* Check for exceptional cases.  */
  if (! M_SUF(__builtin_test_dc_ilogb) (x, 0x7f))
    r = M_SUF (__builtin_ilogb) (x);
  else
    /* Fallback to the generic ilogb if x is NaN, Inf or subnormal.  */
    r = M_SUF (__ieee754_ilogb) (x);
  if (__builtin_expect (r == FP_ILOGB0, 0)
      || __builtin_expect (r == FP_ILOGBNAN, 0)
      || __builtin_expect (r == INT_MAX, 0))
    {
      __set_errno (EDOM);
      __feraiseexcept (FE_INVALID);
    }
  return r;
}
declare_mgen_alias (__ilogb, ilogb)
#else
#include <math/w_ilogb_template.c>
#endif
