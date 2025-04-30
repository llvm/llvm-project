#include <math_ldbl_opt.h>
#include <libm-alias-double.h>
#define __copysign __copysign_generic
#undef libm_alias_double
#define libm_alias_double(a, b)
#undef compat_symbol
#define compat_symbol(a, b, c, d)
#include <sysdeps/sparc/sparc32/fpu/s_copysign.S>
