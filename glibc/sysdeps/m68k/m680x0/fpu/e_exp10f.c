#include <libm-alias-float.h>
#define FUNC __ieee754_exp10f
#define FUNC_FINITE __exp10f
#include <e_acosf.c>
strong_alias (__ieee754_exp10f, __exp10f)
libm_alias_finite (__ieee754_exp10f, __exp10f)
versioned_symbol (libm, __exp10f, exp10f, GLIBC_2_32);
libm_alias_float_other (__ieee754_exp10, exp10)
