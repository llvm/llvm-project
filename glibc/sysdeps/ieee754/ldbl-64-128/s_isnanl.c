#include <math.h>
#include <math_ldbl_opt.h>
#if !IS_IN (libm)
# undef weak_alias
# define weak_alias(n,a)
# undef hidden_def
# define hidden_def(x)
# define __isnanl(arg) ___isnanl(arg)
#endif
#include <sysdeps/ieee754/ldbl-128/s_isnanl.c>
#if !IS_IN (libm)
# undef __isnanl
hidden_ver (___isnanl, __isnanl)
_weak_alias (___isnanl, ____isnanl)
long_double_symbol (libc, ____isnanl, isnanl);
long_double_symbol (libc, ___isnanl, __isnanl);
#endif
