#define USE_AS_MEMMOVE
#define USE_AS_BCOPY
#define MEMCPY		__bcopy_sse2_unaligned
#include "memcpy-sse2-unaligned.S"
