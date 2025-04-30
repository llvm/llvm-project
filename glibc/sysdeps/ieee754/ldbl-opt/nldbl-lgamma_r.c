#include "nldbl-compat.h"

double
attribute_hidden
lgammal_r (double x, int *signgamp)
{
  return lgamma_r (x, signgamp);
}
