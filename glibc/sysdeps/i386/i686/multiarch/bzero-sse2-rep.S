#define USE_AS_BZERO
#define __memset_sse2_rep __bzero_sse2_rep
#include "memset-sse2-rep.S"
