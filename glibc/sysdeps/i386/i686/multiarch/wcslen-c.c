#include <wchar.h>

#if IS_IN (libc)
# define WCSLEN  __wcslen_ia32
#endif

extern __typeof (wcslen) __wcslen_ia32;

#include "wcsmbs/wcslen.c"
