#include <math_ldbl_opt.h>
#undef weak_alias
#define weak_alias(n,a)
#include <sysdeps/ieee754/ldbl-128/s_nexttoward.c>
long_double_symbol (libm, __nexttoward, nexttoward);
