#include <wchar.h>

#define WCSTEST 1
#define L(c) L##c
#define CHAR wchar_t
#define MEMSET wmemset
#define STRLEN wcslen
#define STRNLEN wcsnlen
#define STRCHR wcschr
#define STRRCHR wcsrchr
#define STRCPY wcscpy
#define STRNCPY wcsncpy
#define MEMCMP wmemcmp
#define STPCPY wcpcpy
#define STPNCPY wcpncpy
#define MEMCPY wmemcpy
#define MEMPCPY wmempcpy
#define MEMCHR wmemchr
#define STRCMP wcscmp
#define STRNCMP wcsncmp


#include "../string/stratcliff.c"
