#include "nldbl-compat.h"

long long int
attribute_hidden
llrintl (double x)
{
  return llrint (x);
}
