#include <clc/clc.h>

#define recip(x) (1.0f/x)

#define __CLC_FUNC recip
#define __CLC_BODY <half_unary.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>

#undef recip
