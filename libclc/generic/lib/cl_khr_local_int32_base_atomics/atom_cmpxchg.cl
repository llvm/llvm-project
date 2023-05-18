#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_cmpxchg(volatile local TYPE *p, TYPE cmp, TYPE val) { \
  return atomic_cmpxchg(p, cmp, val); \
}

IMPL(int)
IMPL(unsigned int)
