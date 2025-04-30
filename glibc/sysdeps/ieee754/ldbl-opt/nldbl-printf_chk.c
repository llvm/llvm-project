#include "nldbl-compat.h"

int
attribute_hidden
__printf_chk (int flag, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vfprintf_chk (stdout, flag, fmt, arg);
  va_end (arg);

  return done;
}
