#include <utils.h>

#define __CLC_SW_FUNC(x) __CLC_CONCAT(__clc_, x)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __CLC_SW_FUNC(__CLC_FUNC)(x, y);
}

#undef __CLC_SW_FUNC
