#include "nldbl-compat.h"

int
attribute_hidden
obstack_printf (struct obstack *obstack, const char *fmt, ...)
{
  int result;
  va_list ap;
  va_start (ap, fmt);
  result = __nldbl_obstack_vprintf (obstack, fmt, ap);
  va_end (ap);
  return result;
}
