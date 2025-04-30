#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
csinl (double _Complex x)
{
  return csin (x);
}
