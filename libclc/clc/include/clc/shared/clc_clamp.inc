_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __clc_clamp(__CLC_GENTYPE x,
                                                  __CLC_GENTYPE y,
                                                  __CLC_GENTYPE z);

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __clc_clamp(__CLC_GENTYPE x,
                                                  __CLC_SCALAR_GENTYPE y,
                                                  __CLC_SCALAR_GENTYPE z);
#endif
