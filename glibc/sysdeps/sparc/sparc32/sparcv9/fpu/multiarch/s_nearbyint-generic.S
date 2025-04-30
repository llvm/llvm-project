#include <libm-alias-double.h>
#define __nearbyint __nearbyint_generic
#undef libm_alias_double
#define libm_alias_double(a, b)
#include <sysdeps/sparc/sparc32/sparcv9/fpu/s_nearbyint.S>
