__attribute__((weak)) _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_rsqrt(__CLC_GENTYPE val) {
#pragma clang fp contract(fast)
  return __CLC_FP_LIT(1.0) / __builtin_elementwise_sqrt(val);
}
