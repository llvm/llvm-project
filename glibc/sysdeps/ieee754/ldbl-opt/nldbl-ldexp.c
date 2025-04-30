#include "nldbl-compat.h"

double
attribute_hidden
ldexpl (double x, int exponent)
{
  return ldexp (x, exponent);
}
