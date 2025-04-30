#include "nldbl-compat.h"

int
attribute_hidden
__wprintf_chk (int flag, const wchar_t *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vfwprintf_chk (stdout, flag, fmt, arg);
  va_end (arg);

  return done;
}
