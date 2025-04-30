#include <math-type-macros-float.h>
#include <w_lgamma_template.c>
#if __USE_WRAPPER_TEMPLATE
strong_alias (__lgammaf, __gammaf)
weak_alias (__gammaf, gammaf)
#endif
