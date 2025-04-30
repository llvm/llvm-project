#include "nldbl-compat.h"

int
attribute_hidden
__isoc99_vfscanf (FILE *s, const char *fmt, va_list ap)
{
  return __nldbl___isoc99_vfscanf (s, fmt, ap);
}
