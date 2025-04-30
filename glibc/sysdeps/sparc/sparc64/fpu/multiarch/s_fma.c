#include <sparc-ifunc.h>
#include <math.h>
#include <libm-alias-double.h>

extern __typeof (fma) __fma_vis3 attribute_hidden;
extern __typeof (fma) __fma_generic attribute_hidden;

sparc_libm_ifunc (__fma,
		  hwcap & HWCAP_SPARC_FMAF
		  ? __fma_vis3
		  : __fma_generic);
libm_alias_double (__fma, fma)
