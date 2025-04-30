#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
ccosl (double _Complex x)
{
  return ccos (x);
}
