#include "nldbl-compat.h"

double
attribute_hidden
scalbnl (double x, int n)
{
  return scalbn (x, n);
}
