#define STRCMP __wcsncmp_evex
#define USE_AS_STRNCMP 1
#define USE_AS_WCSCMP 1

#include "strcmp-evex.S"
