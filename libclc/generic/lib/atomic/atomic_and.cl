#include <clc/clc.h>

#define IMPL(TYPE, AS) \
_CLC_OVERLOAD _CLC_DEF TYPE atomic_and(volatile AS TYPE *p, TYPE val) { \
  return __sync_fetch_and_and(p, val); \
}

IMPL(int, global)
IMPL(unsigned int, global)
IMPL(int, local)
IMPL(unsigned int, local)
#undef IMPL
