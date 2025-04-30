#include "nldbl-compat.h"
#include <complex.h>

double
attribute_hidden
cimagl (double _Complex x)
{
  return cimag (x);
}
