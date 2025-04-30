#define USE_AS_STPCPY
#define USE_AS_STRNCPY
#define STRCPY __stpncpy_evex
#include "strcpy-evex.S"
