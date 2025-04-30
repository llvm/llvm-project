#include "nldbl-compat.h"

void
attribute_hidden
syslog (int pri, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  __nldbl_vsyslog (pri, fmt, ap);
  va_end (ap);
}
