#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_dec(local TYPE *p) { \
  return atom_sub(p, (TYPE)1); \
}

IMPL(int)
IMPL(unsigned int)
