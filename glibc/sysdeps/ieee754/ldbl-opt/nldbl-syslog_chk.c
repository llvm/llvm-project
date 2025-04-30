#include "nldbl-compat.h"

void
attribute_hidden
__syslog_chk (int pri, int flag, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  __nldbl___vsyslog_chk (pri, flag, fmt, ap);
  va_end(ap);
}
