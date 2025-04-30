#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
cexpl (double _Complex x)
{
  return cexp (x);
}
