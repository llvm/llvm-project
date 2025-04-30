#include <libm-alias-float.h>
#define __rintf __rintf_generic
#undef libm_alias_float
#define libm_alias_float(a, b)
#include <sysdeps/sparc/sparc32/sparcv9/fpu/s_rintf.S>
