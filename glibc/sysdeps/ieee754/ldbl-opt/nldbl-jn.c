#include "nldbl-compat.h"

double
attribute_hidden
jnl (int n, double x)
{
  return jn (n, x);
}
