#include "nldbl-compat.h"

int
attribute_hidden
strfroml (char *dest, size_t size, const char *format, long double f)
{
  return strfromd (dest, size, format, f);
}
