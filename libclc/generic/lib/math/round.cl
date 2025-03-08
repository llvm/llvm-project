#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_round.h>

#undef __CLC_FUNCTION
#define __CLC_FUNCTION round
#include <clc/math/unary_builtin.inc>
