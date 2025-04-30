#include "nldbl-compat.h"

attribute_hidden
int
__asprintf_chk (char **string_ptr, int flag, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vasprintf_chk (string_ptr, flag, fmt, arg);
  va_end (arg);

  return done;
}
