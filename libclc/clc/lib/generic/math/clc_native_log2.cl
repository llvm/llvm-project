#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_native_log2
#define __CLC_BUILTIN __builtin_elementwise_log2
#include <clc/math/unary_builtin.inc>
