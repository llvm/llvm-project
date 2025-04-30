#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
casinhl (double _Complex x)
{
  return casinh (x);
}
