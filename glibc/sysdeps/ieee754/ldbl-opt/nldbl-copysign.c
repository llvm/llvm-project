#include "nldbl-compat.h"

double
attribute_hidden
copysignl (double x, double y)
{
  return copysign (x, y);
}
