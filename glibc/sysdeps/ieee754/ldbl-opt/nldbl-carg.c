#include "nldbl-compat.h"
#include <complex.h>

double
attribute_hidden
cargl (double _Complex x)
{
  return carg (x);
}
