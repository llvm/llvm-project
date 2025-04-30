#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
cprojl (double _Complex x)
{
  return cproj (x);
}
