#if IS_IN (libc)
# define wcsrchr  __wcsrchr_ia32
#endif

#include "wcsmbs/wcsrchr.c"
