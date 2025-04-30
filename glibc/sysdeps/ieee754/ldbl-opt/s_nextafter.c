#include <math_ldbl_opt.h>
#include <math/s_nextafter.c>
#if LONG_DOUBLE_COMPAT(libm, GLIBC_2_1)
strong_alias (__nextafter, __nexttowardd)
strong_alias (__nextafter, __nexttowardld)
#undef nexttoward
compat_symbol (libm, __nexttowardd, nexttoward, GLIBC_2_1);
compat_symbol (libm, __nexttowardld, nexttowardl, GLIBC_2_1);
#endif
