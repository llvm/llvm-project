/* We have to irritate the compiler a bit.  */
#define ____strtoull_l_internal ____strtoull_l_internal_XXX
#define __strtoull_l __strtoull_l_XXX
#define strtoull_l strtoull_l_XXX

#include <stdlib/strtoul_l.c>

#undef ____strtoull_l_internal
#undef __strtoull_l
#undef strtoull_l
strong_alias (____strtoul_l_internal, ____strtoull_l_internal)
libc_hidden_ver (____strtoul_l_internal, ____strtoull_l_internal)
weak_alias (__strtoul_l, __strtoull_l)
weak_alias (__strtoul_l, strtoull_l)
