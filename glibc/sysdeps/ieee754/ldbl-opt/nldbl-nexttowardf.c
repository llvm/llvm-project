#define nexttowardf nexttowardf_XXX
#include "nldbl-compat.h"
#undef nexttowardf

extern float __nldbl_nexttowardf (float x, double y);

float
attribute_hidden
nexttowardf (float x, double y)
{
  return __nldbl_nexttowardf (x, y);
}
