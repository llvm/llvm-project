#define USE_AS_STPCPY
#define USE_AS_STRNCPY
#define STRCPY __stpncpy_sse2_unaligned
#include "strcpy-sse2-unaligned.S"
