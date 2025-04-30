#include "nldbl-compat.h"

int
attribute_hidden
wprintf (const wchar_t *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vfwprintf (stdout, fmt, arg);
  va_end (arg);

  return done;
}
