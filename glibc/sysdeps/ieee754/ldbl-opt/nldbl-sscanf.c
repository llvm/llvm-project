/* This file defines one of the deprecated scanf variants.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_SCANF
#define __GLIBC_USE_DEPRECATED_SCANF 1

#include "nldbl-compat.h"

int
attribute_hidden
sscanf (const char *s, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vsscanf (s, fmt, arg);
  va_end (arg);

  return done;
}
extern __typeof (sscanf) _IO_sscanf attribute_hidden;
strong_alias (sscanf, _IO_sscanf)
