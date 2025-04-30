#include "nldbl-compat.h"

int
attribute_hidden
__obstack_vprintf_chk (struct obstack *obstack, int flag, const char *fmt,
		       va_list ap)
{
  return __nldbl___obstack_vprintf_chk (obstack, flag, fmt, ap);
}
