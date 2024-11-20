#ifndef __CLC_INTEGER_CLC_ABS_H__
#define __CLC_INTEGER_CLC_ABS_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible abs
#define __clc_abs abs
#else

#define __CLC_BODY <clc/integer/clc_abs.inc>
#include <clc/integer/gentype.inc>

#endif

#endif // __CLC_INTEGER_CLC_ABS_H__
