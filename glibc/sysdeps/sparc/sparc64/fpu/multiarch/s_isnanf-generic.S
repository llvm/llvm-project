#define __isnanf __isnanf_generic
#undef hidden_def
#define hidden_def(a)
#undef weak_alias
#define weak_alias(a,b)
#include <sysdeps/sparc/sparc64/fpu/s_isnanf.S>
