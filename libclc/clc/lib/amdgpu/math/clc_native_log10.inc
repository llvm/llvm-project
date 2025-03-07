_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_native_log10(__CLC_GENTYPE val) {
  return __clc_native_log2(val) * (M_LN2_F / M_LN10_F);
}
