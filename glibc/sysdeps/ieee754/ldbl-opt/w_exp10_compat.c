#include <math_ldbl_opt.h>
#include <math/w_exp10_compat.c>
#if LONG_DOUBLE_COMPAT(libm, GLIBC_2_1)
# if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_27)
strong_alias (__pow10, __pow10_pow10l)
compat_symbol (libm, __pow10_pow10l, pow10l, GLIBC_2_1);
# endif
#endif
