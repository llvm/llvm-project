_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_min(__CLC_GENTYPE a,
                                               __CLC_GENTYPE b) {
  return (b < a ? b : a);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_min(__CLC_GENTYPE a,
                                               __CLC_SCALAR_GENTYPE b) {
  return (b < (__CLC_GENTYPE)a ? (__CLC_GENTYPE)b : a);
}
#endif
