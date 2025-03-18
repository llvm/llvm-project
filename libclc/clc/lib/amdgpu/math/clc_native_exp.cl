#include <clc/float/definitions.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_native_exp2.h>

#define __CLC_BODY <clc_native_exp.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
