#define USE_AS_STPCPY
#define USE_AS_STRNCPY
#define STRCPY __stpncpy_ssse3
#include "strcpy-ssse3.S"
