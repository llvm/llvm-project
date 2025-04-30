#include "nldbl-compat.h"

int
attribute_hidden
__vwprintf_chk (int flag, const wchar_t *fmt, va_list ap)
{
  return __nldbl___vfwprintf_chk (stdout, flag, fmt, ap);
}
