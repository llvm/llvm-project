#ifndef __CLC_SHARED_CLC_MIN_H__
#define __CLC_SHARED_CLC_MIN_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible min
#define __clc_min min
#else

#define __CLC_BODY <clc/shared/clc_min.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/shared/clc_min.inc>
#include <clc/math/gentype.inc>

#endif

#endif // __CLC_SHARED_CLC_MIN_H__
