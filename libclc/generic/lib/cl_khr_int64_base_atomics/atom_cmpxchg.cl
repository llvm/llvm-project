#include <clc/clc.h>

#ifdef cl_khr_int64_base_atomics

#define IMPL(AS, TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_cmpxchg(volatile AS TYPE *p, TYPE cmp, TYPE val) { \
  return __sync_val_compare_and_swap_8(p, cmp, val); \
}

IMPL(global, long)
IMPL(global, unsigned long)
IMPL(local, long)
IMPL(local, unsigned long)
#undef IMPL

#endif
