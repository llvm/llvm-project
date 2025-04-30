#include <math_ldbl_opt.h>
#include <sysdeps/ieee754/dbl-64/s_copysign.c>
#if LONG_DOUBLE_COMPAT (libc, GLIBC_2_0)
compat_symbol (libc, __copysign, copysignl, GLIBC_2_0);
#endif
