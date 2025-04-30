#define __signbitl __signbitl_XXX
#include "nldbl-compat.h"
#undef __signbitl

int
attribute_hidden
__signbitl (double x)
{
  return signbit (x);
}
