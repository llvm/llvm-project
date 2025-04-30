#include "nldbl-compat.h"

double
attribute_hidden
powl (double x, double y)
{
  return pow (x, y);
}
