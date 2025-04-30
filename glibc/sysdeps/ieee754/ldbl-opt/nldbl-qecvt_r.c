#define qecvt_r qecvt_r_XXX
#include "nldbl-compat.h"
#undef qecvt_r

int
attribute_hidden
qecvt_r (double val, int ndigit, int *__restrict decpt, int *__restrict sign,
	 char *__restrict buf, size_t len)
{
  return ecvt_r (val, ndigit, decpt, sign, buf, len);
}
