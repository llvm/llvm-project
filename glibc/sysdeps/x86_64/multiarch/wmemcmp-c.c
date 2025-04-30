#if IS_IN (libc)
# include <wchar.h>

# define WMEMCMP  __wmemcmp_sse2

extern __typeof (wmemcmp) __wmemcmp_sse2;
#endif

#include "wcsmbs/wmemcmp.c"
