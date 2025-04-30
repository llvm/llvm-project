#include <libm-alias-float.h>
#define __fabsf __fabsf_generic
#undef libm_alias_float
#define libm_alias_float(a, b)
#include <sysdeps/sparc/sparc32/fpu/s_fabsf.S>
