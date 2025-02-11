_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE __clc_abs_diff(__CLC_GENTYPE x,
                                                      __CLC_GENTYPE y) {
  __CLC_U_GENTYPE ux = __builtin_astype(x, __CLC_U_GENTYPE);
  __CLC_U_GENTYPE uy = __builtin_astype(y, __CLC_U_GENTYPE);
  return x > y ? ux - uy : uy - ux;
}
