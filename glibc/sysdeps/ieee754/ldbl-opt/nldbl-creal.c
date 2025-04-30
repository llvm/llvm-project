#include "nldbl-compat.h"
#include <complex.h>

double
attribute_hidden
creall (double _Complex x)
{
  return creal (x);
}
