#include "nldbl-compat.h"

void
attribute_hidden
sincosl (double x, double *sinx, double *cosx)
{
  sincos (x, sinx, cosx);
}
