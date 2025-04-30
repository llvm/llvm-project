#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
csqrtl (double _Complex x)
{
  return csqrt (x);
}
