#include <clc/clc.h>

#ifdef cl_khr_int64_base_atomics

#define IMPL(AS, TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_xchg(volatile AS TYPE *p, TYPE val) { \
  return __sync_swap_8(p, val); \
}

IMPL(global, long)
IMPL(global, unsigned long)
IMPL(local, long)
IMPL(local, unsigned long)
#undef IMPL

#endif
