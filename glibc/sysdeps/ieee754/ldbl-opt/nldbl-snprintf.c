#include "nldbl-compat.h"

int
attribute_hidden
weak_function
snprintf (char *s, size_t maxlen, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vsnprintf (s, maxlen, fmt, arg);
  va_end (arg);

  return done;
}
