#include "nldbl-compat.h"
#include <complex.h>

double
attribute_hidden
cabsl (double _Complex x)
{
  return cabs (x);
}
