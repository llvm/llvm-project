_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE min(__CLC_GENTYPE a, __CLC_GENTYPE b) {
  return (a < b ? a : b);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE min(__CLC_GENTYPE a, __CLC_SCALAR_GENTYPE b) {
  return (a < (__CLC_GENTYPE)b ? a : (__CLC_GENTYPE)b);
}
#endif
