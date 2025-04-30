#include <float128_private.h>
#define SIG 0
#define FUNC __setpayloadf128
#include "../ldbl-128/s_setpayloadl_main.c"
libm_alias_float128 (__setpayload, setpayload)
