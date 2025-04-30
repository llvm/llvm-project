#include "nldbl-compat.h"

attribute_hidden
void
vsyslog (int pri, const char *fmt, va_list ap)
{
  __nldbl_vsyslog (pri, fmt, ap);
}
