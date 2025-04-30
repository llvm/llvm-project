#include <math_ldbl_opt.h>
#include <math/w_lgamma_compat.c>
#if LONG_DOUBLE_COMPAT(libm, GLIBC_2_0)
strong_alias (__lgamma_compat, __lgammal_dbl_compat)
compat_symbol (libm, __lgammal_dbl_compat, lgammal, GLIBC_2_0);
compat_symbol (libm, __gamma, gammal, GLIBC_2_0);
#endif
