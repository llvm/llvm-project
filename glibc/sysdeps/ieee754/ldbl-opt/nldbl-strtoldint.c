#define __strtold_internal __strtold_internal_XXX
#include "nldbl-compat.h"
#undef __strtold_internal

double
attribute_hidden
__strtold_internal (const char *nptr, char **endptr, int group)
{
  return __strtod_internal (nptr, endptr, group);
}
