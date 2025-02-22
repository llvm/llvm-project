_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE mix(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                         __CLC_GENTYPE a) {
  return __clc_mad(y - x, a, x);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE mix(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                         __CLC_SCALAR_GENTYPE a) {
  return mix(x, y, (__CLC_GENTYPE)a);
}
#endif
