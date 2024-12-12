#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fabs.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION fabs
#include <clc/math/unary_builtin.inc>
