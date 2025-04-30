#include <math_ldbl_opt.h>
#include <sysdeps/ieee754/dbl-64/s_finite.c>
weak_alias (__finite, ___finite)
#if IS_IN (libm)
# if LONG_DOUBLE_COMPAT(libm, GLIBC_2_1)
compat_symbol (libm, __finite, __finitel, GLIBC_2_1);
# endif
# if LONG_DOUBLE_COMPAT(libm, GLIBC_2_0)
compat_symbol (libm, ___finite, finitel, GLIBC_2_0);
# endif
#else
# if LONG_DOUBLE_COMPAT(libc, GLIBC_2_0)
compat_symbol (libm, __finite, __finitel, GLIBC_2_0);
# endif
# if LONG_DOUBLE_COMPAT(libc, GLIBC_2_0)
compat_symbol (libc, ___finite, finitel, GLIBC_2_0);
# endif
#endif
