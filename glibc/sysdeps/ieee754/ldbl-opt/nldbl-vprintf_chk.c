#include "nldbl-compat.h"

int
attribute_hidden
__vprintf_chk (int flag, const char *fmt, va_list ap)
{
  return __nldbl___vfprintf_chk (stdout, flag, fmt, ap);
}
