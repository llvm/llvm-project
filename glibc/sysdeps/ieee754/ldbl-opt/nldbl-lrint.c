#include "nldbl-compat.h"

long int
attribute_hidden
lrintl (double x)
{
  return lrint (x);
}
