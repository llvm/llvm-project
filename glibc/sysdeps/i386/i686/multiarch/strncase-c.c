#include <string.h>

extern __typeof (strncasecmp) __strncasecmp_nonascii;

#define __strncasecmp __strncasecmp_nonascii
#include <string/strncase.c>

strong_alias (__strncasecmp_nonascii, __strncasecmp_ia32)
