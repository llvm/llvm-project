#ifndef MEMCHR
# define MEMCHR __memchr_evex_rtm
#endif

#define USE_IN_RTM 1
#define SECTION(p) p##.evex.rtm

#include "memchr-evex.S"
