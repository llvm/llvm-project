#ifndef __CLC_RELATIONAL_CLC_ISLESSEQUAL_H__
#define __CLC_RELATIONAL_CLC_ISLESSEQUAL_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible islessequal
#define __clc_islessequal islessequal
#else

#define __CLC_FUNCTION __clc_islessequal
#define __CLC_BODY <clc/relational/binary_decl.inc>

#include <clc/relational/floatn.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

#endif

#endif // __CLC_RELATIONAL_CLC_ISLESSEQUAL_H__
