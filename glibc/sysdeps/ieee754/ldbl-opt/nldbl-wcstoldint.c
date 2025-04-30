#define __wcstold_internal __wcstold_internal_XXX
#include "nldbl-compat.h"
#undef __wcstold_internal

double
attribute_hidden
__wcstold_internal (const wchar_t *nptr, wchar_t **endptr, int group)
{
  return __wcstod_internal (nptr, endptr, group);
}
