#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_floor
#define __CLC_BUILTIN __builtin_elementwise_floor
#include <clc/math/unary_builtin.inc>
