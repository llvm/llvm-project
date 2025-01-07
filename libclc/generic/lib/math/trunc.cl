#include <clc/clc.h>
#include <clc/math/clc_trunc.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION trunc
#include <clc/math/unary_builtin.inc>
