#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
ccoshl (double _Complex x)
{
  return ccosh (x);
}
