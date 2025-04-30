#define USE_AS_WMEMCHR 1
#define wmemchr __wmemchr_sse2

#include "../memchr.S"
