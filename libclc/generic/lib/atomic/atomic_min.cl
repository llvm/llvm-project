#include <clc/clc.h>

#define IMPL(TYPE, AS, OP) \
_CLC_OVERLOAD _CLC_DEF TYPE atomic_min(volatile AS TYPE *p, TYPE val) { \
  return __sync_fetch_and_##OP(p, val); \
}

IMPL(int, global, min)
IMPL(unsigned int, global, umin)
IMPL(int, local, min)
IMPL(unsigned int, local, umin)
#undef IMPL
