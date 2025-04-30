#include "nldbl-compat.h"

int
attribute_hidden
weak_function
fwprintf (FILE *stream, const wchar_t *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vfwprintf (stream, fmt, arg);
  va_end (arg);

  return done;
}
