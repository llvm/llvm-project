#include <clc/clc.h>
#include <clc/math/clc_native_sqrt.h>

#define __FLOAT_ONLY
#define __CLC_FUNCTION native_sqrt
#include <clc/math/unary_builtin.inc>
