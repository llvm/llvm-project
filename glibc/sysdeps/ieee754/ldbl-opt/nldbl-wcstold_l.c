#define wcstold_l wcstold_l_XXX
#define __wcstold_l __wcstold_l_XXX
#include "nldbl-compat.h"
#undef wcstold_l
#undef __wcstold_l

double
attribute_hidden
__wcstold_l (const wchar_t *nptr, wchar_t **endptr, locale_t loc)
{
  return __wcstod_l (nptr, endptr, loc);
}
extern __typeof (__wcstold_l) wcstold_l attribute_hidden;
weak_alias (__wcstold_l, wcstold_l)
