#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
conjl (double _Complex x)
{
  return conj (x);
}
