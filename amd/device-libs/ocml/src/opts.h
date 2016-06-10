
#include "oclc.h"

#define HAVE_FAST_FMA32() __oclc_have_fast_fma32()
#define HAVE_FAST_FMA64() __oclc_have_fast_fma64()
#define FINITE_ONLY_OPT() __oclc_finite_only_opt()
#define FAST_RELAXED_OPT() __oclc_fast_relaxed_opt()
#define DAZ_OPT() __oclc_daz_opt()
#define CORRECTLY_ROUNDED_SQRT32() __oclc_correctly_rounded_sqrt32()
#define AMD_OPT() __oclc_amd_opt()

