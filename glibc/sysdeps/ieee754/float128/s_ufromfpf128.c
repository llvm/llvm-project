#define UNSIGNED 1
#define INEXACT 0
#define FUNC __ufromfpf128
#include <float128_private.h>
#include "../ldbl-128/s_fromfpl_main.c"
libm_alias_float128 (__ufromfp, ufromfp)
