/* We have to irritate the compiler a bit.  */
#define ____wcstoull_l_internal ____wcstoull_l_internal_XXX
#define __wcstoull_l ___wcstoull_l_XXX
#define wcstoull_l __wcstoull_l_XXX

#include <wcsmbs/wcstoul_l.c>

#undef ____wcstoull_l_internal
#undef __wcstoull_l
#undef wcstoull_l
strong_alias (____wcstoul_l_internal, ____wcstoull_l_internal)
weak_alias (__wcstoul_l, __wcstoull_l)
weak_alias (__wcstoul_l, wcstoull_l)
