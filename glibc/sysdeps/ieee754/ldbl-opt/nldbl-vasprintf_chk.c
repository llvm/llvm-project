#include "nldbl-compat.h"

int
attribute_hidden
__vasprintf_chk (char **result_ptr, int flag, const char *fmt, va_list ap)
{
  return __nldbl___vasprintf_chk (result_ptr, flag, fmt, ap);
}
