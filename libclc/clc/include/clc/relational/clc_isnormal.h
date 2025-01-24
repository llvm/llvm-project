#ifndef __CLC_RELATIONAL_CLC_ISNORMAL_H__
#define __CLC_RELATIONAL_CLC_ISNORMAL_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible isnormal
#define __clc_isnormal isnormal
#else

#define __CLC_FUNCTION __clc_isnormal
#define __CLC_BODY <clc/relational/unary_decl.inc>

#include <clc/relational/floatn.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

#endif

#endif // __CLC_RELATIONAL_CLC_ISNORMAL_H__
