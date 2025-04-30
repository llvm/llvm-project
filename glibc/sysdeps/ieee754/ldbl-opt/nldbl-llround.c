#include "nldbl-compat.h"

long long int
attribute_hidden
llroundl (double x)
{
  return llround (x);
}
