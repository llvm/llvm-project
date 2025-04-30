#if IS_IN (libc)
# define WCSCPY  __wcscpy_sse2
#endif

#include <wcsmbs/wcscpy.c>
