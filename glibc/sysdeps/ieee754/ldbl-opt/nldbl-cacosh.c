#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
cacoshl (double _Complex x)
{
  return cacosh (x);
}
