#define USE_AS_MEMMOVE
#define MEMCPY		__memmove_sse2_unaligned
#define MEMCPY_CHK	__memmove_chk_sse2_unaligned
#include "memcpy-sse2-unaligned.S"
