#include <utils.h>

#define __CLC_SW_FUNC(x) __CLC_CONCAT(__clc_, x)

// TODO: Enable half precision when the sw routine is implemented
#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __CLC_SW_FUNC(__CLC_FUNC)(x, y);
}
#endif

#undef __CLC_SW_FUNC
