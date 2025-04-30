#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
csinhl (double _Complex x)
{
  return csinh (x);
}
