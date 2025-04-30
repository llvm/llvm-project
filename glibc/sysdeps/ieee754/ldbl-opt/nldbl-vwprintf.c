#include "nldbl-compat.h"

int
attribute_hidden
vwprintf (const wchar_t *fmt, va_list ap)
{
  return __nldbl_vfwprintf (stdout, fmt, ap);
}
