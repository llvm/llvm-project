#define __lrint __lrint_generic
#undef weak_alias
#define weak_alias(a,b)
#undef strong_alias
#define strong_alias(a,b)
#include <sysdeps/sparc/sparc64/fpu/s_lrint.S>
