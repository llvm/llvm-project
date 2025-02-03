#ifndef __CLC_RELATIONAL_CLC_ALL_H__
#define __CLC_RELATIONAL_CLC_ALL_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible all
#define __clc_all all
#else

#include <clc/clcfunc.h>
#include <clc/clctypes.h>

#define _CLC_ALL_DECL(TYPE) _CLC_OVERLOAD _CLC_DECL int __clc_all(TYPE v);

#define _CLC_VECTOR_ALL_DECL(TYPE)                                             \
  _CLC_ALL_DECL(TYPE)                                                          \
  _CLC_ALL_DECL(TYPE##2)                                                       \
  _CLC_ALL_DECL(TYPE##3)                                                       \
  _CLC_ALL_DECL(TYPE##4)                                                       \
  _CLC_ALL_DECL(TYPE##8)                                                       \
  _CLC_ALL_DECL(TYPE##16)

_CLC_VECTOR_ALL_DECL(char)
_CLC_VECTOR_ALL_DECL(short)
_CLC_VECTOR_ALL_DECL(int)
_CLC_VECTOR_ALL_DECL(long)

#undef _CLC_ALL_DECL
#undef _CLC_VECTOR_ALL_DECL

#endif

#endif // __CLC_RELATIONAL_CLC_ALL_H__
