#include "nldbl-compat.h"

attribute_hidden
int
fprintf (FILE *stream, const char *fmt, ...)
{
  va_list arg;
  int done;

  va_start (arg, fmt);
  done = __nldbl_vfprintf (stream, fmt, arg);
  va_end (arg);

  return done;
}
extern __typeof (fprintf) _IO_fprintf attribute_hidden;
weak_alias (fprintf, _IO_fprintf)
