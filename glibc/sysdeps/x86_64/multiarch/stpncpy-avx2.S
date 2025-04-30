#define USE_AS_STPCPY
#define USE_AS_STRNCPY
#define STRCPY __stpncpy_avx2
#include "strcpy-avx2.S"
