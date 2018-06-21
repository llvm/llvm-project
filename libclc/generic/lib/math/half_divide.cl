#include <clc/clc.h>

#define divide(x,y) (x/y)

#define __CLC_FUNC divide
#define __CLC_BODY <half_binary.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
#undef divide
