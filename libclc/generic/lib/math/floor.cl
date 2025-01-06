#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_floor.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION floor
#include <clc/math/unary_builtin.inc>
