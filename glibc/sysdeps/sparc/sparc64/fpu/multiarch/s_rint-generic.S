#define __rint __rint_generic
#undef weak_alias
#define weak_alias(a, b)
#include <sysdeps/sparc/sparc64/fpu/s_rint.S>
