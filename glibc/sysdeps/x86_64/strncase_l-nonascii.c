#include <string.h>

extern int __strncasecmp_l_nonascii (const char *__s1, const char *__s2,
				     size_t __n, locale_t __loc);

#define __strncasecmp_l __strncasecmp_l_nonascii
#define USE_IN_EXTENDED_LOCALE_MODEL    1
#include <string/strncase.c>
