#if IS_IN (libc)
# include <wchar.h>

# define WCSNLEN __wcsnlen_sse2

extern __typeof (wcsnlen) __wcsnlen_sse2;
#endif

#include "wcsmbs/wcsnlen.c"
