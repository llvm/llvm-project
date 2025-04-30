#include "nldbl-compat.h"

int
attribute_hidden
__isoc99_vsscanf (const char *string, const char *fmt, va_list ap)
{
  return __nldbl___isoc99_vsscanf (string, fmt, ap);
}
