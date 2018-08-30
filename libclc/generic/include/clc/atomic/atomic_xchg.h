#define __CLC_FUNCTION atomic_xchg

_CLC_OVERLOAD _CLC_DECL float __CLC_FUNCTION (volatile local float *, float);
_CLC_OVERLOAD _CLC_DECL float __CLC_FUNCTION (volatile global float *, float);
#include <clc/atomic/atomic_decl.inc>
