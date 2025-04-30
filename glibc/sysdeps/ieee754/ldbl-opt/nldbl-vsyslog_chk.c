#include "nldbl-compat.h"

void
attribute_hidden
__vsyslog_chk (int pri, int flag, const char *fmt, va_list ap)
{
  __nldbl___vsyslog_chk (pri, flag, fmt, ap);
}
