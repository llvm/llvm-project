#include <math.h>
#include <math_ldbl_opt.h>
#undef weak_alias
#define weak_alias(n,a)
#undef hidden_def
#define hidden_def(x)
#define __finitel(arg) ___finitel(arg)
#include <sysdeps/ieee754/ldbl-128/s_finitel.c>
#undef __finitel
hidden_ver (___finitel, __finitel)
_weak_alias (___finitel, ____finitel)
#if IS_IN (libm)
long_double_symbol (libm, ____finitel, finitel);
long_double_symbol (libm, ___finitel, __finitel);
#else
long_double_symbol (libc, ____finitel, finitel);
long_double_symbol (libc, ___finitel, __finitel);
#endif
