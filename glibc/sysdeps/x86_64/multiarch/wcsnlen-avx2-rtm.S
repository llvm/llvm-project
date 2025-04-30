#define STRLEN __wcsnlen_avx2_rtm
#define USE_AS_WCSLEN 1
#define USE_AS_STRNLEN 1

#include "strlen-avx2-rtm.S"
