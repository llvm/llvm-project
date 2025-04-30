#define USE_AS_STPCPY
#define USE_AS_STRNCPY
#define STRCPY __stpncpy_sse2
#include "strcpy-sse2.S"
