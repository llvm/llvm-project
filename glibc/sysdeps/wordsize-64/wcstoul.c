/* We have to irritate the compiler a bit.  */
#define __wcstoull_internal __wcstoull_internal_XXX
#define wcstoull wcstoull_XXX
#define wcstouq wcstouq_XXX

#include <wcsmbs/wcstoul.c>

#undef __wcstoull_internal
#undef wcstoull
#undef wcstouq
strong_alias (__wcstoul_internal, __wcstoull_internal)
libc_hidden_ver (__wcstoul_internal, __wcstoull_internal)
weak_alias (wcstoul, wcstoull)
weak_alias (wcstoul, wcstouq)
weak_alias (wcstoul, wcstoumax)
