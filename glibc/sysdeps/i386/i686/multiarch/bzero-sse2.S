#define USE_AS_BZERO
#define __memset_sse2 __bzero_sse2
#include "memset-sse2.S"
