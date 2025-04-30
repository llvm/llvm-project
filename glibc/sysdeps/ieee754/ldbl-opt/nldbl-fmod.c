#include "nldbl-compat.h"

double
attribute_hidden
fmodl (double x, double y)
{
  return fmod (x, y);
}
