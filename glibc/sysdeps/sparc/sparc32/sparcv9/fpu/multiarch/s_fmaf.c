#include <sparc-ifunc.h>
#include <math.h>
#include <libm-alias-float.h>

extern float __fmaf_vis3 (float, float, float);
extern float __fmaf_generic (float, float, float);

sparc_libm_ifunc(__fmaf, hwcap & HWCAP_SPARC_FMAF ? __fmaf_vis3 : __fmaf_generic);
libm_alias_float (__fma, fma)
