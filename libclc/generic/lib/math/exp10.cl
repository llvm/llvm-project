#include <clc/clc.h>
#include <math/clc_exp10.h>

#define __CLC_FUNC exp10
#define __CLC_BODY <clc_sw_unary.inc>
#include <clc/math/gentype.inc>
