_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE abs_diff(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __builtin_astype((__CLC_GENTYPE)(x > y ? x-y : y-x), __CLC_U_GENTYPE);
}
