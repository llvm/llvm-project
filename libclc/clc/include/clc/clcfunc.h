//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
#define _CLC_GENERIC_AS_SUPPORTED 1
// Note that we hard-code the assumption that a non-distinct address space means
// that the target maps the generic address space to the private address space.
#ifdef __CLC_DISTINCT_GENERIC_ADDRSPACE__
#define _CLC_DISTINCT_GENERIC_AS_SUPPORTED 1
#else
#define _CLC_DISTINCT_GENERIC_AS_SUPPORTED 0
#endif
#else
#define _CLC_GENERIC_AS_SUPPORTED 0
#define _CLC_DISTINCT_GENERIC_AS_SUPPORTED 0
#endif

#endif // __CLC_CLCFUNC_H_
