/* We have to irritate the compiler a bit.  */
#define __strtoll_internal __strtoll_internal_XXX
#define strtoll strtoll_XXX
#define strtoq strtoq_XXX

#include <stdlib/strtol.c>

#undef __strtoll_internal
#undef strtoll
#undef strtoq
strong_alias (__strtol_internal, __strtoll_internal)
libc_hidden_ver (__strtol_internal, __strtoll_internal)
weak_alias (strtol, strtoll)
libc_hidden_ver (strtol, strtoll)
weak_alias (strtol, strtoq)
libc_hidden_ver (strtol, strtoq)
weak_alias (strtol, strtoimax)
