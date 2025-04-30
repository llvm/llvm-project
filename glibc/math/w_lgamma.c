#include <math-type-macros-double.h>
#include <w_lgamma_template.c>
#if __USE_WRAPPER_TEMPLATE
strong_alias (__lgamma, __gamma)
weak_alias (__gamma, gamma)
# ifdef NO_LONG_DOUBLE
strong_alias (__gamma, __gammal)
weak_alias (__gammal, gammal)
# endif
#endif
