#include "nldbl-compat.h"

int
attribute_hidden
vfprintf (FILE *s, const char *fmt, va_list ap)
{
  return __nldbl_vfprintf (s, fmt, ap);
}
extern __typeof (vfprintf) _IO_vfprintf attribute_hidden;
strong_alias (vfprintf, _IO_vfprintf)
