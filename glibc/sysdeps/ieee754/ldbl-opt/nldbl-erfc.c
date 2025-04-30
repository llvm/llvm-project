#include "nldbl-compat.h"

double
attribute_hidden
erfcl (double x)
{
  return erfc (x);
}
