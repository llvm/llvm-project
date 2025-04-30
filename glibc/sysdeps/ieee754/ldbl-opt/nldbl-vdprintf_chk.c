#include "nldbl-compat.h"

int
attribute_hidden
__vdprintf_chk (int d, int flag, const char *fmt, va_list arg)
{
  return __nldbl___vdprintf_chk (d, flag, fmt, arg);
}
