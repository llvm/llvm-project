#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
catanl (double _Complex x)
{
  return catan (x);
}
