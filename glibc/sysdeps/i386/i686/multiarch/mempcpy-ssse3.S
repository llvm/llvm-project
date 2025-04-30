#define USE_AS_MEMPCPY
#define MEMCPY		__mempcpy_ssse3
#define MEMCPY_CHK	__mempcpy_chk_ssse3
#include "memcpy-ssse3.S"
