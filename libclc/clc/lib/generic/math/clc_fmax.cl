#include <clc/internal/clc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION __clc_fmax
#define __CLC_BUILTIN __builtin_elementwise_max
#include <clc/math/binary_builtin.inc>
