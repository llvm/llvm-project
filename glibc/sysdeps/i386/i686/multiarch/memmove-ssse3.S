#define USE_AS_MEMMOVE
#define MEMCPY		__memmove_ssse3
#define MEMCPY_CHK	__memmove_chk_ssse3
#include "memcpy-ssse3.S"
