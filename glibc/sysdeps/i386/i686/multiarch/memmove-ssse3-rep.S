#define USE_AS_MEMMOVE
#define MEMCPY		__memmove_ssse3_rep
#define MEMCPY_CHK	__memmove_chk_ssse3_rep
#include "memcpy-ssse3-rep.S"
