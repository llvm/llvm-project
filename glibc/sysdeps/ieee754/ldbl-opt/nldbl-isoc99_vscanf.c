#include "nldbl-compat.h"

int
attribute_hidden
__isoc99_vscanf (const char *fmt, va_list ap)
{
  return __nldbl___isoc99_vfscanf (stdin, fmt, ap);
}
