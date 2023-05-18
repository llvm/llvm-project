#include <clc/clc.h>

#include "../../../generic/lib/clcmacro.h"

_CLC_OVERLOAD _CLC_DEF float native_rsqrt(float x)
{
    return __builtin_r600_recipsqrt_ieeef(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, native_rsqrt, float);
