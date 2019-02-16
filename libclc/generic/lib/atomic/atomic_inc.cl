#include <clc/clc.h>

#define IMPL(TYPE, AS) \
_CLC_OVERLOAD _CLC_DEF TYPE atomic_inc(volatile AS TYPE *p) { \
  return __sync_fetch_and_add(p, (TYPE)1); \
}

IMPL(int, global)
IMPL(unsigned int, global)
IMPL(int, local)
IMPL(unsigned int, local)
#undef IMPL
