#include "nldbl-compat.h"

double
attribute_hidden
modfl (double x, double *iptr)
{
  return modf (x, iptr);
}
