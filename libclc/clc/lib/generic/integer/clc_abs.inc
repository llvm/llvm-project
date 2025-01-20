_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE __clc_abs(__CLC_GENTYPE x) {
  return __builtin_astype((__CLC_GENTYPE)(x > (__CLC_GENTYPE)(0) ? x : -x),
                          __CLC_U_GENTYPE);
}
