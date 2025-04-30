#include "nldbl-compat.h"

int
attribute_hidden
__vfwprintf_chk (FILE *s, int flag, const wchar_t *fmt, va_list ap)
{
  return __nldbl___vfwprintf_chk (s, flag, fmt, ap);
}
