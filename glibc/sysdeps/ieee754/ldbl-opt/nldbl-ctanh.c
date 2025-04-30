#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
ctanhl (double _Complex x)
{
  return ctanh (x);
}
