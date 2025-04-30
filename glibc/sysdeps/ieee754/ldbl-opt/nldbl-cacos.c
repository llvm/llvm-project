#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
cacosl (double _Complex x)
{
  return cacos (x);
}
