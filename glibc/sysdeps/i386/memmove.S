#define USE_AS_MEMMOVE
#define MEMCPY		memmove
#define MEMCPY_CHK	__memmove_chk
#include "memcpy.S"
