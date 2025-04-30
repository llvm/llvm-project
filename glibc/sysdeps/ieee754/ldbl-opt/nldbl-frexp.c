#include "nldbl-compat.h"

double
attribute_hidden
frexpl (double x, int *exponent)
{
  return frexp (x, exponent);
}
