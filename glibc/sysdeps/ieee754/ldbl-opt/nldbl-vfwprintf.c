#include "nldbl-compat.h"

int
attribute_hidden
weak_function
vfwprintf (FILE *s, const wchar_t *fmt, va_list ap)
{
  return __nldbl_vfwprintf (s, fmt, ap);
}
