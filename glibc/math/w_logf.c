#include <math-type-macros-float.h>
#undef __USE_WRAPPER_TEMPLATE
#define __USE_WRAPPER_TEMPLATE 1
#undef declare_mgen_alias
#define declare_mgen_alias(a, b)
#include <w_log_template.c>
versioned_symbol (libm, __logf, logf, GLIBC_2_27);
libm_alias_float_other (__log, log)
