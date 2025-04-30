#include "nldbl-compat.h"

int
attribute_hidden
weak_function
vswprintf (wchar_t *string, size_t maxlen, const wchar_t *fmt, va_list ap)
{
  return __nldbl_vswprintf (string, maxlen, fmt, ap);
}
