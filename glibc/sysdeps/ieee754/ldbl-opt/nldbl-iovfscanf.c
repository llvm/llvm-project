/* This file defines one of the deprecated scanf variants.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_SCANF
#define __GLIBC_USE_DEPRECATED_SCANF 1

#include "nldbl-compat.h"

int
attribute_hidden
_IO_vfscanf (FILE *s, const char *fmt, va_list ap, int *errp)
{
  return __nldbl__IO_vfscanf (s, fmt, ap, errp);
}
