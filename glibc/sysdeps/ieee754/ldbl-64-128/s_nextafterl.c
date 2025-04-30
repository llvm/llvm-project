#include <math_ldbl_opt.h>
#undef weak_alias
#define weak_alias(n,a)
#include <sysdeps/ieee754/ldbl-128/s_nextafterl.c>
long_double_symbol (libm, __nextafterl, nextafterl);
long_double_symbol (libm, __nexttowardl, nexttowardl);
#undef weak_alias
#define weak_alias(name, aliasname) _weak_alias (name, aliasname)
libm_alias_ldouble_other (__nextafter, nextafter)
