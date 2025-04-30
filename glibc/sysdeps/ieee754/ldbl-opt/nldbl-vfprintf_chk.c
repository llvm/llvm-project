#include "nldbl-compat.h"

int
attribute_hidden
__vfprintf_chk (FILE *s, int flag, const char *fmt, va_list ap)
{
  return __nldbl___vfprintf_chk (s, flag, fmt, ap);
}
