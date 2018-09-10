#include <clc/clc.h>

#define IMPL(TYPE, AS) \
_CLC_OVERLOAD _CLC_DEF TYPE atomic_cmpxchg(volatile AS TYPE *p, TYPE cmp, TYPE val) { \
  return __sync_val_compare_and_swap(p, cmp, val); \
}

IMPL(int, global)
IMPL(unsigned int, global)
IMPL(int, local)
IMPL(unsigned int, local)
#undef IMPL
