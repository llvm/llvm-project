#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_min(local TYPE *p, TYPE val) { \
  return atomic_min(p, val); \
}

IMPL(int)
IMPL(unsigned int)
