#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_fmin
#define __CLC_BUILTIN __builtin_elementwise_min
#include <clc/math/binary_builtin.inc>
