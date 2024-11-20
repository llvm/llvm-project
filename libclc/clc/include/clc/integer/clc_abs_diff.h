#ifndef __CLC_INTEGER_CLC_ABS_DIFF_H__
#define __CLC_INTEGER_CLC_ABS_DIFF_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible abs_diff
#define __clc_abs_diff abs_diff
#else

#define __CLC_BODY <clc/integer/clc_abs_diff.inc>
#include <clc/integer/gentype.inc>

#endif

#endif // __CLC_INTEGER_CLC_ABS_DIFF_H__
