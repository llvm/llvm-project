/* Workaround PR90731 with GCC 9 when using ldbl redirects in C++.  */
#include <bits/floatn.h>
#if defined __cplusplus && __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
# if __GNUC_PREREQ (9, 0) && !__GNUC_PREREQ (9, 3)
#   pragma GCC system_header
# endif
#endif

#include <stdlib/monetary.h>
#ifndef _ISOMAC
#include <stdarg.h>

extern ssize_t
__vstrfmon_l_internal (char *s, size_t maxsize, locale_t loc,
		       const char *format, va_list ap,
		       unsigned int flags)
  attribute_hidden;

/* Flags for __vstrfmon_l_internal.

   STRFMON_LDBL_IS_DBL is a one-bit mask for the flags parameter that
   indicates whether long double values are to be handled as having the
   same format as double, in which case the flag should be set to one,
   or as another format, otherwise.  */
#define STRFMON_LDBL_IS_DBL 0x0001
#define STRFMON_LDBL_USES_FLOAT128  0x0002

#endif
