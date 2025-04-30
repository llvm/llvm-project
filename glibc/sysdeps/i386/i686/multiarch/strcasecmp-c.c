#include <string.h>

extern __typeof (strcasecmp) __strcasecmp_nonascii;

#define __strcasecmp __strcasecmp_nonascii
#include <string/strcasecmp.c>

strong_alias (__strcasecmp_nonascii, __strcasecmp_ia32)

/* The needs of strcasecmp in libc are minimal, no need to go through
   the IFUNC.  */
strong_alias (__strcasecmp_nonascii, __GI___strcasecmp)
