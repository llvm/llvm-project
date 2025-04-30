#define AS_WCSLEN
#define strlen	__wcslen_sse4_1

#include "strlen-vec.S"
