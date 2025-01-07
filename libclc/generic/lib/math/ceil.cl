#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_ceil.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION ceil
#include <clc/math/unary_builtin.inc>
