#include <math-type-macros-ldouble.h>
#include <w_lgamma_template.c>
#if __USE_WRAPPER_TEMPLATE
strong_alias (__lgammal, __gammal)
weak_alias (__gammal, gammal)
#endif
