#include "nldbl-compat.h"

int
attribute_hidden
ilogbl (double x)
{
  return ilogb (x);
}
