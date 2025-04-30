#include "nldbl-compat.h"

int
attribute_hidden
weak_function
vasprintf (char **result_ptr, const char *fmt, va_list ap)
{
  return __nldbl_vasprintf (result_ptr, fmt, ap);
}
