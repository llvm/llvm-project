#define __signbit __signbit_generic
#undef strong_alias
#define strong_alias(a, b)
#undef hidden_def
#define hidden_def(a)
#include <sysdeps/sparc/sparc64/fpu/s_signbit.S>
