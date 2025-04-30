#include "nldbl-compat.h"

ssize_t
attribute_hidden
__strfmon_l (char *s, size_t maxsize, locale_t loc, const char *format, ...)
{
  va_list ap;
  ssize_t res;

  va_start (ap, format);
  res = __nldbl___vstrfmon_l (s, maxsize, loc, format, ap);
  va_end (ap);
  return res;
}
extern __typeof (__strfmon_l) strfmon_l attribute_hidden;
weak_alias (__strfmon_l, strfmon_l)
