#include <libm-alias-double.h>
#define __llrint __llrint_generic
#undef libm_alias_double
#define libm_alias_double(a,b)
#include <sysdeps/sparc/sparc32/sparcv9/fpu/s_llrint.S>
