#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF float atomic_xchg(volatile global float *p, float val) {
  return as_float(atomic_xchg((volatile global int *)p, as_int(val)));
}

_CLC_OVERLOAD _CLC_DEF float atomic_xchg(volatile local float *p, float val) {
  return as_float(atomic_xchg((volatile local int *)p, as_int(val)));
}
