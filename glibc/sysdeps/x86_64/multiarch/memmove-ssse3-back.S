#define USE_AS_MEMMOVE
#define MEMCPY		__memmove_ssse3_back
#define MEMCPY_CHK	__memmove_chk_ssse3_back
#include "memcpy-ssse3-back.S"
