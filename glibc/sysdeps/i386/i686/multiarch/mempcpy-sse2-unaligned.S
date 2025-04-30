#define USE_AS_MEMPCPY
#define MEMCPY		__mempcpy_sse2_unaligned
#define MEMCPY_CHK	__mempcpy_chk_sse2_unaligned
#include "memcpy-sse2-unaligned.S"
