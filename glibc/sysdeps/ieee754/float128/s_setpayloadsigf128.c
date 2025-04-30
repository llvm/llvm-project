#include <float128_private.h>
#define SIG 1
#define FUNC __setpayloadsigf128
#include "../ldbl-128/s_setpayloadl_main.c"
libm_alias_float128 (__setpayloadsig, setpayloadsig)
