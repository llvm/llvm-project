#include <libm-alias-ldouble.h>

long double __fabsl (long double x)
{
  return __builtin_fabsl (x);
}
libm_alias_ldouble (__fabs, fabs)
