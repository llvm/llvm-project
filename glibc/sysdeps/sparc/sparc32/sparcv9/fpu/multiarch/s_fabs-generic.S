#include <libm-alias-double.h>
#define __fabs __fabs_generic
#undef libm_alias_double
#define libm_alias_double(a, b)
#include <sysdeps/sparc/sparc32/fpu/s_fabs.S>
