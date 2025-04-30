#define USE_AS_STPCPY
#define USE_AS_STRNCPY
#define STRCPY __stpncpy_avx2_rtm
#include "strcpy-avx2-rtm.S"
