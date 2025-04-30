#include "nldbl-compat.h"

double
attribute_hidden
scalbl (double x, double n)
{
  return scalb (x, n);
}
