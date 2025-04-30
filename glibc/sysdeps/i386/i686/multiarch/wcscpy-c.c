#if IS_IN (libc)
# define wcscpy  __wcscpy_ia32
#endif

#include "wcsmbs/wcscpy.c"
