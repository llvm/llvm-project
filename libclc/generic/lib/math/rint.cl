#include <clc/clc.h>
#include <clc/math/clc_rint.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION rint
#include <clc/math/unary_builtin.inc>
