#include "nldbl-compat.h"

int
attribute_hidden
__snprintf_chk (char *s, size_t maxlen, int flag, size_t slen,
		const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vsnprintf_chk (s, maxlen, flag, slen, fmt, arg);
  va_end (arg);

  return done;
}
