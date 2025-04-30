#include "nldbl-compat.h"

int
attribute_hidden
printf (const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vfprintf (stdout, fmt, arg);
  va_end (arg);

  return done;
}
extern __typeof (printf) _IO_printf attribute_hidden;
strong_alias (printf, _IO_printf)
