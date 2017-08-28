#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_and(global TYPE *p, TYPE val) { \
  return atomic_and(p, val); \
}

IMPL(int)
IMPL(unsigned int)