#ifndef MEMCHR
# define MEMCHR __memchr_avx2_rtm
#endif

#define ZERO_UPPER_VEC_REGISTERS_RETURN \
  ZERO_UPPER_VEC_REGISTERS_RETURN_XTEST

#define VZEROUPPER_RETURN jmp	 L(return_vzeroupper)

#define SECTION(p) p##.avx.rtm

#include "memchr-avx2.S"
