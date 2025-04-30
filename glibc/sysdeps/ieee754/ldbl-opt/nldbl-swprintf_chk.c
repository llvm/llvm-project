#include "nldbl-compat.h"

int
attribute_hidden
__swprintf_chk (wchar_t *s, size_t n, int flag, size_t slen,
		const wchar_t *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl___vswprintf_chk (s, n, flag, slen, fmt, arg);
  va_end (arg);

  return done;
}
