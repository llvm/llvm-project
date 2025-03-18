#include <clc/clc.h>
#include <clc/math/clc_native_cos.h>

#define __FLOAT_ONLY
#define __CLC_FUNCTION native_cos
#include <clc/math/unary_builtin.inc>
