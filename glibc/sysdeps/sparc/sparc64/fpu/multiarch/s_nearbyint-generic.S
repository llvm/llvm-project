#define __nearbyint __nearbyint_generic
#undef weak_alias
#define weak_alias(a,b)
#include <sysdeps/sparc/sparc64/fpu/s_nearbyint.S>
