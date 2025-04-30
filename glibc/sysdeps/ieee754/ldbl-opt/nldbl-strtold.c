#define strtold strtold_XXX
#include "nldbl-compat.h"
#undef strtold

double
attribute_hidden
strtold (const char *nptr, char **endptr)
{
  return strtod (nptr, endptr);
}
