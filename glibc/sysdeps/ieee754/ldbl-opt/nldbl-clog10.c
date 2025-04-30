#include "nldbl-compat.h"
#include <complex.h>

double _Complex
attribute_hidden
clog10l (double _Complex x)
{
  return clog10 (x);
}
extern __typeof (clog10l) __clog10l attribute_hidden;
weak_alias (clog10l, __clog10l)
