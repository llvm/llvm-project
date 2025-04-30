#include "nldbl-compat.h"

double
attribute_hidden
nearbyintl (double x)
{
  return nearbyint (x);
}
