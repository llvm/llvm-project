#include "nldbl-compat.h"

int
attribute_hidden
__vsnprintf_chk (char *string, size_t maxlen, int flag, size_t slen,
		 const char *fmt, va_list ap)
{
  return __nldbl___vsnprintf_chk (string, maxlen, flag, slen, fmt, ap);
}
