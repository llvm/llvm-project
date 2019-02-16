#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_dec(volatile global TYPE *p) { \
  return atomic_dec(p); \
}

IMPL(int)
IMPL(unsigned int)
