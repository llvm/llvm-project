#include "nldbl-compat.h"

double
attribute_hidden
significandl (double x)
{
  return significand (x);
}
