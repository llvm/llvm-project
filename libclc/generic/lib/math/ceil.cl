#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_ceil.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION ceil
#include "unary_builtin.inc"
