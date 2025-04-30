#define USE_FMA_BUILTIN 1
#define USE_FMAF_BUILTIN 1
#define USE_FMAL_BUILTIN 0
/* This is not available for P8 or BE targets.  */
#ifdef __FP_FAST_FMAF128
# define USE_FMAF128_BUILTIN 1
#else
# define USE_FMAF128_BUILTIN 0
#endif
