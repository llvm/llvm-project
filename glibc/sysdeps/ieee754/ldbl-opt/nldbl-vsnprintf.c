#include "nldbl-compat.h"

int
attribute_hidden
vsnprintf (char *string, size_t maxlen, const char *fmt, va_list ap)
{
  return __nldbl_vsnprintf (string, maxlen, fmt, ap);
}
extern __typeof (vsnprintf) __vsnprintf attribute_hidden;
weak_alias (vsnprintf, __vsnprintf)
