#include <math_ldbl_opt.h>
#include <libm-alias-ldouble.h>
#if IS_IN (libc)
# undef libm_alias_ldouble
# define libm_alias_ldouble(from, to)
#endif
#include <sysdeps/ieee754/ldbl-128/s_modfl.c>
#if IS_IN (libc)
long_double_symbol (libc, __modfl, modfl);
#endif
