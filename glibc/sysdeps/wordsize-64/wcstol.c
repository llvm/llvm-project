/* We have to irritate the compiler a bit.  */
#define __wcstoll_internal __wcstoll_internal_XXX
#define wcstoll wcstoll_XXX
#define wcstoq wcstoq_XXX

#include <wcsmbs/wcstol.c>

#undef __wcstoll_internal
#undef wcstoll
#undef wcstoq
strong_alias (__wcstol_internal, __wcstoll_internal)
libc_hidden_ver (__wcstol_internal, __wcstoll_internal)
weak_alias (wcstol, wcstoll)
weak_alias (wcstol, wcstoq)
weak_alias (wcstol, wcstoimax)
