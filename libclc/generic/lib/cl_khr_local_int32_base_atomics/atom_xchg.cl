#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_xchg(local TYPE *p, TYPE val) { \
  return atomic_xchg(p, val); \
}

IMPL(int)
IMPL(unsigned int)
