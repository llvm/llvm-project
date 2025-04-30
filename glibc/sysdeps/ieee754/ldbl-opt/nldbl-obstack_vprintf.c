#include "nldbl-compat.h"

int
attribute_hidden
obstack_vprintf (struct obstack *obstack, const char *fmt, va_list ap)
{
  return __nldbl_obstack_vprintf (obstack, fmt, ap);
}
