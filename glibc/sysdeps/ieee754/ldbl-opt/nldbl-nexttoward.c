#define nexttoward nexttoward_XXX
#define nexttowardl nexttowardl_XXX
#include "nldbl-compat.h"
#undef nexttoward
#undef nexttowardl

double
attribute_hidden
nexttoward (double x, double y)
{
  return nextafter (x, y);
}
extern __typeof (nexttoward) nexttowardl attribute_hidden;
strong_alias (nexttoward, nexttowardl)
