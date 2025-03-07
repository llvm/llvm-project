#include <clc/clc.h>
#include <clc/math/clc_native_sin.h>

#define __FLOAT_ONLY
#define __CLC_FUNCTION native_sin
#include <clc/math/unary_builtin.inc>
