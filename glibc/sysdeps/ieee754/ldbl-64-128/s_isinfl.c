#include <math.h>
#include <math_ldbl_opt.h>
#if !IS_IN (libm)
# undef weak_alias
# define weak_alias(n,a)
# undef hidden_def
# define hidden_def(x)
# define __isinfl(arg) ___isinfl(arg)
#endif
#include <sysdeps/ieee754/ldbl-128/s_isinfl.c>
#if !IS_IN (libm)
# undef __isinfl
hidden_ver (___isinfl, __isinfl)
_weak_alias (___isinfl, ____isinfl)
long_double_symbol (libc, ____isinfl, isinfl);
long_double_symbol (libc, ___isinfl, __isinfl);
#endif
