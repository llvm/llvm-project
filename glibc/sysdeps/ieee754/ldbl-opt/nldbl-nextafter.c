#include "nldbl-compat.h"

double
attribute_hidden
nextafterl (double x, double y)
{
  return nextafter (x, y);
}
