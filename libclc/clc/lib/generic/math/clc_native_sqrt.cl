#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_native_sqrt
#define __CLC_BUILTIN __builtin_elementwise_sqrt
#include <clc/math/unary_builtin.inc>
