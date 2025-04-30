#include "nldbl-compat.h"

double
attribute_hidden
atan2l (double x, double y)
{
  return atan2 (x, y);
}
