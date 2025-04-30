#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
casinl (double _Complex x)
{
  return casin (x);
}
