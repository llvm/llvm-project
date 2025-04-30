#include "nldbl-compat.h"

int
attribute_hidden
__vswprintf_chk (wchar_t *string, size_t maxlen, int flag, size_t slen,
		 const wchar_t *fmt, va_list ap)
{
  return __nldbl___vswprintf_chk (string, maxlen, flag, slen, fmt, ap);
}
