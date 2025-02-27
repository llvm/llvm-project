_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_fma(__CLC_GENTYPE a, __CLC_GENTYPE b,
                                               __CLC_GENTYPE c) {
#if __CLC_FPSIZE == 32
  if (!__CLC_HAVE_HW_FMA32())
    return __clc_sw_fma(a, b, c);
#endif
  return __builtin_elementwise_fma(a, b, c);
}
