#define wcstold wcstold_XXX
#include "nldbl-compat.h"
#undef wcstold

double
attribute_hidden
wcstold (const wchar_t *nptr, wchar_t **endptr)
{
  return wcstod (nptr, endptr);
}
