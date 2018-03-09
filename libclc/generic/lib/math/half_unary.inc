#include <utils.h>

#define __CLC_HALF_FUNC(x) __CLC_CONCAT(half_, x)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_HALF_FUNC(__CLC_FUNC)(__CLC_GENTYPE val) {
  return __CLC_FUNC(val);
}

#undef __CLC_HALF_FUNC
