#include "nldbl-compat.h"

ssize_t
attribute_hidden
strfmon (char *s, size_t maxsize, const char *format, ...)
{
  va_list ap;
  ssize_t res;

  va_start (ap, format);
  res = __nldbl___vstrfmon (s, maxsize, format, ap);
  va_end (ap);
  return res;
}
