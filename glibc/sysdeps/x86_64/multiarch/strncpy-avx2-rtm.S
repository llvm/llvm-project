#define USE_AS_STRNCPY
#define STRCPY __strncpy_avx2_rtm
#include "strcpy-avx2-rtm.S"
