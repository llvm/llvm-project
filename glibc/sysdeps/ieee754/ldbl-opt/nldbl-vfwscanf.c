/* This file defines one of the deprecated scanf variants.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_SCANF
#define __GLIBC_USE_DEPRECATED_SCANF 1

#include "nldbl-compat.h"

int
attribute_hidden
vfwscanf (FILE *s, const wchar_t *fmt, va_list ap)
{
  return __nldbl_vfwscanf (s, fmt, ap);
}
