#include "nldbl-compat.h"

double
attribute_hidden
remquol (double x, double y, int *quo)
{
  return remquo (x, y, quo);
}
