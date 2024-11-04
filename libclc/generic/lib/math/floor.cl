#include "../clcmacro.h"
#include <clc/clc.h>
#include <clc/math/clc_floor.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION floor
#include "unary_builtin.inc"
