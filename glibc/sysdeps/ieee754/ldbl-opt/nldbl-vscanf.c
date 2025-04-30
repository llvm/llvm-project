/* This file defines one of the deprecated scanf variants.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_SCANF
#define __GLIBC_USE_DEPRECATED_SCANF 1

#include "nldbl-compat.h"

int
attribute_hidden
weak_function
vscanf (const char *fmt, va_list ap)
{
  return __nldbl__IO_vfscanf (stdin, fmt, ap, NULL);
}
