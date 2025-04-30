#include "nldbl-compat.h"

attribute_hidden
int
__dprintf_chk (int d, int flag, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vdprintf_chk (d, flag, fmt, arg);
  va_end (arg);

  return done;
}
