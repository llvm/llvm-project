/* Skip the optimization for long double as ibm128 does not provide an
   optimized builtin. */
#include <math-type-macros-ldouble.h>
#include <math/w_ilogb_template.c>
