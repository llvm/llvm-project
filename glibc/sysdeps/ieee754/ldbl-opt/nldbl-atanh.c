#include "nldbl-compat.h"

double
attribute_hidden
atanhl (double x)
{
  return atanh (x);
}
