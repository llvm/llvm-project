#include <math.h>
#include <math_ldbl_opt.h>
#undef weak_alias
#define weak_alias(n,a)
#define __fpclassifyl ___fpclassifyl
#undef libm_hidden_def
#define libm_hidden_def(a)
#include <sysdeps/ieee754/ldbl-128/s_fpclassifyl.c>
#undef __fpclassifyl
long_double_symbol (libm, ___fpclassifyl, __fpclassifyl);
libm_hidden_ver (___fpclassifyl, __fpclassifyl)
