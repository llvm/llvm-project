#include "nldbl-compat.h"

int
attribute_hidden
swprintf (wchar_t *s, size_t n, const wchar_t *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vswprintf (s, n, fmt, arg);
  va_end (arg);

  return done;
}
