#ifndef __CLC_RELATIONAL_CLC_SELECT_H__
#define __CLC_RELATIONAL_CLC_SELECT_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible select
#define __clc_select select
#else

/* Duplciate these so we don't have to distribute utils.h */
#define __CLC_CONCAT(x, y) x##y
#define __CLC_XCONCAT(x, y) __CLC_CONCAT(x, y)

#define __CLC_BODY <clc/relational/clc_select.inc>
#include <clc/math/gentype.inc>
#define __CLC_BODY <clc/relational/clc_select.inc>
#include <clc/integer/gentype.inc>

#undef __CLC_CONCAT
#undef __CLC_XCONCAT

#endif

#endif // __CLC_RELATIONAL_CLC_SELECT_H__
