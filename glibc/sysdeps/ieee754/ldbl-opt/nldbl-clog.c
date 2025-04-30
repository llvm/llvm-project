#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
clogl (double _Complex x)
{
  return clog (x);
}
