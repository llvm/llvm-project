#include <math-type-macros-double.h>
#include <w_remainder_template.c>
#if __USE_WRAPPER_TEMPLATE
weak_alias (__remainder, drem)
# ifdef NO_LONG_DOUBLE
weak_alias (__remainder, dreml)
# endif
#endif
