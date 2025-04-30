#include <libm-alias-float.h>

float __fabsf (float x)
{
  return __builtin_fabsf (x);
}
libm_alias_float (__fabs, fabs)
