#define USE_AS_MEMPCPY
#define MEMCPY		__mempcpy_ssse3_rep
#define MEMCPY_CHK	__mempcpy_chk_ssse3_rep
#include "memcpy-ssse3-rep.S"
