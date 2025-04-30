#include "nldbl-compat.h"

int
attribute_hidden
vprintf (const char *fmt, va_list ap)
{
  return __nldbl_vfprintf (stdout, fmt, ap);
}
