#define strtold_l strtold_l_XXX
#define __strtold_l __strtold_l_XXX
#define __strtod_l __strtod_l_XXX
#include "nldbl-compat.h"
#undef strtold_l
#undef __strtold_l
#undef __strtod_l

extern double
__strtod_l (const char *__restrict __nptr, char **__restrict __endptr,
	    locale_t __loc);

double
attribute_hidden
__strtold_l (const char *nptr, char **endptr, locale_t loc)
{
  return __strtod_l (nptr, endptr, loc);
}
extern __typeof (__strtold_l) strtold_l attribute_hidden;
weak_alias (__strtold_l, strtold_l)
