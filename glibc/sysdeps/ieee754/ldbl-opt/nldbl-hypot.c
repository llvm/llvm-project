#include "nldbl-compat.h"

double
attribute_hidden
hypotl (double x, double y)
{
  return hypot (x, y);
}
