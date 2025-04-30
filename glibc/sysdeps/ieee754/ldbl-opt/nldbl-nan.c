#include "nldbl-compat.h"

double
attribute_hidden
nanl (const char *tag)
{
  return nan (tag);
}
