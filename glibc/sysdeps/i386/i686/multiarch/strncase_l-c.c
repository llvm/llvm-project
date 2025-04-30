#include <string.h>

extern __typeof (strncasecmp_l) __strncasecmp_l_nonascii;

#define __strncasecmp_l __strncasecmp_l_nonascii
#define USE_IN_EXTENDED_LOCALE_MODEL    1
#include <string/strncase.c>

strong_alias (__strncasecmp_l_nonascii, __strncasecmp_l_ia32)

/* The needs of strcasecmp in libc are minimal, no need to go through
   the IFUNC.  */
strong_alias (__strncasecmp_l_nonascii, __GI___strncasecmp_l)
