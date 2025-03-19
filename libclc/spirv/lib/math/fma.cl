#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/internal/math/clc_sw_fma.h>

_CLC_DEFINE_TERNARY_BUILTIN(float, fma, __clc_sw_fma, float, float, float)
