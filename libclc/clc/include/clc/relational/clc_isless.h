#ifndef __CLC_RELATIONAL_CLC_ISLESS_H__
#define __CLC_RELATIONAL_CLC_ISLESS_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible isless
#define __clc_isless isless
#else

#define __CLC_FUNCTION __clc_isless
#define __CLC_BODY <clc/relational/binary_decl.inc>

#include <clc/relational/floatn.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

#endif

#endif // __CLC_RELATIONAL_CLC_ISLESS_H__
