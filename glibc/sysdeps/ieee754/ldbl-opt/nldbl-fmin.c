#include "nldbl-compat.h"

double
attribute_hidden
fminl (double x, double y)
{
  return fmin (x, y);
}
