#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_inc(volatile global TYPE *p) { \
  return atomic_inc(p); \
}

IMPL(int)
IMPL(unsigned int)
