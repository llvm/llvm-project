#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_native_exp
#define __CLC_BUILTIN __builtin_elementwise_exp
#include <clc/math/unary_builtin.inc>
