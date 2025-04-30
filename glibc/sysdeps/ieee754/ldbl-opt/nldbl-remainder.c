#include "nldbl-compat.h"

double
attribute_hidden
remainderl (double x, double y)
{
  return remainder (x, y);
}
extern __typeof (remainderl) dreml attribute_hidden;
weak_alias (remainderl, dreml)
