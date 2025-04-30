#include "nldbl-compat.h"

double
attribute_hidden
fmal (double x, double y, double z)
{
  return fma (x, y, z);
}
