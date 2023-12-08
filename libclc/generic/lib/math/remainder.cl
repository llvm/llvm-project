#include <clc/clc.h>
#include <math/clc_remainder.h>

#define __CLC_FUNC remainder
#define __CLC_BODY <clc_sw_binary.inc>
#include <clc/math/gentype.inc>
