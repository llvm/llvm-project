#include <sparc-ifunc.h>
#include <math.h>
#include <math_ldbl_opt.h>
#include <libm-alias-double.h>

extern double __fma_vis3 (double, double, double);
extern double __fma_generic (double, double, double);

sparc_libm_ifunc(__fma, hwcap & HWCAP_SPARC_FMAF ? __fma_vis3 : __fma_generic);
libm_alias_double (__fma, fma)
