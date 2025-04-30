#include <libm-alias-double.h>

double __fabs (double x)
{
  return __builtin_fabs (x);
}
libm_alias_double (__fabs, fabs)
