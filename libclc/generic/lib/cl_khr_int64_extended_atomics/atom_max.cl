#include <clc/clc.h>

#ifdef cl_khr_int64_extended_atomics

unsigned long __clc__sync_fetch_and_max_local_8(volatile local long *, long);
unsigned long __clc__sync_fetch_and_max_global_8(volatile global long *, long);
unsigned long __clc__sync_fetch_and_umax_local_8(volatile local unsigned long *, unsigned long);
unsigned long __clc__sync_fetch_and_umax_global_8(volatile global unsigned long *, unsigned long);

#define IMPL(AS, TYPE, OP) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_max(volatile AS TYPE *p, TYPE val) { \
  return __clc__sync_fetch_and_##OP##_##AS##_8(p, val); \
}

IMPL(global, long, max)
IMPL(global, unsigned long, umax)
IMPL(local, long, max)
IMPL(local, unsigned long, umax)
#undef IMPL

#endif
