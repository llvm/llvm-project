_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE max(__CLC_GENTYPE a, __CLC_GENTYPE b) {
  return __clc_max(a, b);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE max(__CLC_GENTYPE a,
                                         __CLC_SCALAR_GENTYPE b) {
  return __clc_max(a, b);
}
#endif
