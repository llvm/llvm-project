#define USE_AS_MEMMOVE
#define USE_AS_BCOPY
#define MEMCPY		__bcopy_ssse3_rep
#include "memcpy-ssse3-rep.S"
