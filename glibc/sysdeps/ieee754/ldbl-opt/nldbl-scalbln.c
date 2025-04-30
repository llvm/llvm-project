#include "nldbl-compat.h"

double
attribute_hidden
scalblnl (double x, long int n)
{
  return scalbln (x, n);
}
