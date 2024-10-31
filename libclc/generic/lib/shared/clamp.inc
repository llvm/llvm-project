_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE clamp(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_GENTYPE z) {
  return __clc_clamp(x, y, z);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE clamp(__CLC_GENTYPE x, __CLC_SCALAR_GENTYPE y, __CLC_SCALAR_GENTYPE z) {
  return __clc_clamp(x, y, z);
}
#endif
