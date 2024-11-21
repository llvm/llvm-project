#ifndef __CLC_SHARED_CLC_MAX_H__
#define __CLC_SHARED_CLC_MAX_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible max
#define __clc_max max
#else

#define __CLC_BODY <clc/shared/clc_max.inc>
#include <clc/integer/gentype.inc>

#define __CLC_BODY <clc/shared/clc_max.inc>
#include <clc/math/gentype.inc>

#endif

#endif // __CLC_SHARED_CLC_MAX_H__
