#define ZERO_UPPER_VEC_REGISTERS_RETURN \
  ZERO_UPPER_VEC_REGISTERS_RETURN_XTEST

#define VZEROUPPER_RETURN jmp	 L(return)

#define SECTION(p) p##.avx.rtm
#define MEMSET_SYMBOL(p,s)	p##_avx2_##s##_rtm
#define WMEMSET_SYMBOL(p,s)	p##_avx2_##s##_rtm

#include "memset-avx2-unaligned-erms.S"
