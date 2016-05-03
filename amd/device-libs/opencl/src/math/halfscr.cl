
#include "llvm.h"

__attribute__((always_inline)) float
__half_scr(float x, __private float *cp)
{
    float ax = fabs(x);
    *cp = __llvm_amdgcn_cos_f32(x);
    float s = __llvm_amdgcn_sin_f32(x);
    return ax < 0x1.0p-50f ? x : s;
}

