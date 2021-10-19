#include <clc/clc.h>
#include <math/clc_fma.h>

#define __CLC_BODY <fma.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>

bool __clc_runtime_has_hw_fma32()
{
    return false;
}
