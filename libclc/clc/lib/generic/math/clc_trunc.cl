#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_trunc
#define __CLC_BUILTIN __builtin_elementwise_trunc
#include <clc/math/unary_builtin.inc>
