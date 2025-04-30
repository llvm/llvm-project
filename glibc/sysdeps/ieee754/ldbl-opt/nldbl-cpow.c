#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
cpowl (double _Complex x, double _Complex y)
{
  return cpow (x, y);
}
