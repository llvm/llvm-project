#include <utils.h>

#define __CLC_SW_FUNC(x) __CLC_CONCAT(__clc_, x)

#if __CLC_FPSIZE > 16
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __CLC_SW_FUNC(__CLC_FUNC)(x, y);
}
#elif __CLC_FPSIZE == 16
#ifdef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x,
                                                __CLC_GENTYPE y) {
  return convert_half(
      __CLC_SW_FUNC(__CLC_FUNC)(convert_float(x), convert_float(y)));
}
#else
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x,
                                                __CLC_GENTYPE y) {
  return __CLC_XCONCAT(convert_half, __CLC_VECSIZE)(__CLC_SW_FUNC(__CLC_FUNC)(
      __CLC_XCONCAT(convert_float, __CLC_VECSIZE)(x),
      __CLC_XCONCAT(convert_float, __CLC_VECSIZE)(y)));
}
#endif
#endif

#undef __CLC_SW_FUNC
