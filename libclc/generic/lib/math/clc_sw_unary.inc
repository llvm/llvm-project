#include <clc/utils.h>

#define __CLC_SW_FUNC(x) __CLC_CONCAT(__clc_, x)

#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return __CLC_SW_FUNC(__CLC_FUNC)(x);
}
#elif __CLC_FPSIZE == 16
#ifdef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return convert_half(__CLC_SW_FUNC(__CLC_FUNC)(convert_float(x)));
}
#else
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return __CLC_XCONCAT(convert_half, __CLC_VECSIZE)(__CLC_SW_FUNC(__CLC_FUNC)(
      __CLC_XCONCAT(convert_float, __CLC_VECSIZE)(x)));
}
#endif
#endif

#undef __CLC_SW_FUNC
