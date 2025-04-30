#include <string.h>

extern __typeof (strcasecmp_l) __strcasecmp_l_nonascii;

#define __strcasecmp_l __strcasecmp_l_nonascii
#define USE_IN_EXTENDED_LOCALE_MODEL    1
#include <string/strcasecmp.c>

strong_alias (__strcasecmp_l_nonascii, __strcasecmp_l_ia32)

/* The needs of strcasecmp in libc are minimal, no need to go through
   the IFUNC.  */
strong_alias (__strcasecmp_l_nonascii, __GI___strcasecmp_l)
