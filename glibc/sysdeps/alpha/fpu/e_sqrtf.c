#include <shlib-compat.h>

#include <sysdeps/ieee754/flt-32/e_sqrtf.c>

#if SHLIB_COMPAT (libm, GLIBC_2_18, GLIBC_2_31)
strong_alias (__ieee754_sqrtf, __sqrtf_finite_2_18)
compat_symbol (libm, __sqrtf_finite_2_18, __sqrtf_finite, GLIBC_2_18);
#endif
