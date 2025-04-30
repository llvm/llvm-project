#include "nldbl-compat.h"

int
attribute_hidden
__sprintf_chk (char *s, int flag, size_t slen, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vsprintf_chk (s, flag, slen, fmt, arg);
  va_end (arg);

  return done;
}
