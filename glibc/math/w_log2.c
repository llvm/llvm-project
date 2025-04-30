#include <math-type-macros-double.h>
#undef __USE_WRAPPER_TEMPLATE
#define __USE_WRAPPER_TEMPLATE 1
#undef declare_mgen_alias
#define declare_mgen_alias(a, b)
#include <w_log2_template.c>
versioned_symbol (libm, __log2, log2, GLIBC_2_29);
libm_alias_double_other (__log2, log2)
