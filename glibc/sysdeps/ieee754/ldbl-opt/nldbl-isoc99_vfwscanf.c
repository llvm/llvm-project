#include "nldbl-compat.h"

int
attribute_hidden
__isoc99_vfwscanf (FILE *s, const wchar_t *fmt, va_list ap)
{
  return __nldbl___isoc99_vfwscanf (s, fmt, ap);
}
