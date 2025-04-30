#include "nldbl-compat.h"

int
attribute_hidden
__obstack_printf_chk (struct obstack *obstack, int flag, const char *fmt, ...)
{
  int result;
  va_list ap;
  va_start (ap, fmt);
  result = __nldbl___obstack_vprintf_chk (obstack, flag, fmt, ap);
  va_end (ap);
  return result;
}
