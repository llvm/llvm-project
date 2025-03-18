#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_native_sin
#define __CLC_BUILTIN __builtin_elementwise_sin
#include <clc/math/unary_builtin.inc>
