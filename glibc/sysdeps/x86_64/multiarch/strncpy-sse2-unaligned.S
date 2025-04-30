#define USE_AS_STRNCPY
#define STRCPY __strncpy_sse2_unaligned
#include "strcpy-sse2-unaligned.S"
