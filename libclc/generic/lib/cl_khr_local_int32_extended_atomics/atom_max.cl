#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_max(local TYPE *p, TYPE val) { \
  return atomic_max(p, val); \
}

IMPL(int)
IMPL(unsigned int)
