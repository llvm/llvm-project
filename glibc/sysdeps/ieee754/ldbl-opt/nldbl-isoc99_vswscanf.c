#include "nldbl-compat.h"

int
attribute_hidden
__isoc99_vswscanf (const wchar_t *string, const wchar_t *fmt, va_list ap)
{
  return __nldbl___isoc99_vswscanf (string, fmt, ap);
}
