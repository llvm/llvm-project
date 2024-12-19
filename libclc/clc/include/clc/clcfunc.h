#ifndef __CLC_CLCFUNC_H_
#define __CLC_CLCFUNC_H_

#define _CLC_OVERLOAD __attribute__((overloadable))
#define _CLC_DECL
#define _CLC_INLINE __attribute__((always_inline)) inline

// avoid inlines for SPIR-V related targets since we'll optimise later in the
// chain
#if defined(CLC_SPIRV)
#define _CLC_DEF
#elif defined(CLC_CLSPV)
#define _CLC_DEF __attribute__((noinline)) __attribute__((clspv_libclc_builtin))
#else
#define _CLC_DEF __attribute__((always_inline))
#endif

#endif // __CLC_CLCFUNC_H_
