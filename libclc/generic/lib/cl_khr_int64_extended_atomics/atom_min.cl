#include <clc/clc.h>

#ifdef cl_khr_int64_extended_atomics

unsigned long __clc__sync_fetch_and_min_local_8(volatile local long *, long);
unsigned long __clc__sync_fetch_and_min_global_8(volatile global long *, long);
unsigned long __clc__sync_fetch_and_umin_local_8(volatile local unsigned long *, unsigned long);
unsigned long __clc__sync_fetch_and_umin_global_8(volatile global unsigned long *, unsigned long);

#define IMPL(AS, TYPE, OP) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_min(volatile AS TYPE *p, TYPE val) { \
  return __clc__sync_fetch_and_##OP##_##AS##_8(p, val); \
}

IMPL(global, long, min)
IMPL(global, unsigned long, umin)
IMPL(local, long, min)
IMPL(local, unsigned long, umin)
#undef IMPL

#endif
