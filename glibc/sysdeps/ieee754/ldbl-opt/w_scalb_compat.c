#include <math_ldbl_opt.h>
#include <math/w_scalb_compat.c>
#if LIBM_SVID_COMPAT
# if LONG_DOUBLE_COMPAT(libm, GLIBC_2_0)
compat_symbol (libm, __scalb, scalbl, GLIBC_2_0);
# endif
#endif
