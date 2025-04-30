#include <math_ldbl_opt.h>
#undef weak_alias
#define weak_alias(n,a)
#define USE_AS_COMPAT 1
#include <math/lgamma-compat.h>
#undef LGAMMA_OLD_VER
#define LGAMMA_OLD_VER LONG_DOUBLE_COMPAT_VERSION
#include <math/w_lgamma_compatl.c>
#if GAMMA_ALIAS
long_double_symbol (libm, __gammal, gammal);
#endif
