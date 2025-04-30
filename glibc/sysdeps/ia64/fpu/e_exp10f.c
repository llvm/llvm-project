/* IA64 does not provided the finite symbol alias.  */
#include <libm-alias-finite.h>
#undef libm_alias_finite
#define libm_alias_finite(a, b)
#include <sysdeps/ieee754/flt-32/e_exp10f.c>
