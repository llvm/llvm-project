#define qecvt qecvt_XXX
#include "nldbl-compat.h"
#undef qecvt

attribute_hidden
char *
qecvt (double val, int ndigit, int *__restrict decpt, int *__restrict sign)
{
  return ecvt (val, ndigit, decpt, sign);
}
