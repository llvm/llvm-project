#define qfcvt_r qfcvt_r_XXX
#include "nldbl-compat.h"
#undef qfcvt_r

int
attribute_hidden
qfcvt_r (double val, int ndigit, int *__restrict decpt, int *__restrict sign,
	 char *__restrict buf, size_t len)
{
  return fcvt_r (val, ndigit, decpt, sign, buf, len);
}
