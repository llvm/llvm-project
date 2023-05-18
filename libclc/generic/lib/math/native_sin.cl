#include <clc/clc.h>

#define __CLC_NATIVE_INTRINSIC sin

#define __CLC_BODY <native_unary_intrinsic.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
