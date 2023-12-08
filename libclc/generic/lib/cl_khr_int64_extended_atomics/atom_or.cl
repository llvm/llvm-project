#include <clc/clc.h>

#ifdef cl_khr_int64_extended_atomics

#define IMPL(AS, TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_or(volatile AS TYPE *p, TYPE val) { \
  return __sync_fetch_and_or_8(p, val); \
}

IMPL(global, long)
IMPL(global, unsigned long)
IMPL(local, long)
IMPL(local, unsigned long)
#undef IMPL

#endif
