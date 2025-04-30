#include "nldbl-compat.h"

int
attribute_hidden
__isnanl (double x)
{
  return isnan (x);
}
extern __typeof (__isnanl) isnanl attribute_hidden;
weak_alias (__isnanl, isnanl)
