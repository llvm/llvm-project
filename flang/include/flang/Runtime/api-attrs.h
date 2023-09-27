/*===-- include/flang/Runtime/api-attrs.h ---------------------------*- C -*-=//
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 */

/*
 * The file defines a set macros that can be used to apply
 * different attributes/pragmas to functions/variables
 * declared/defined/used in Flang runtime library.
 */

#ifndef FORTRAN_RUNTIME_API_ATTRS_H_
#define FORTRAN_RUNTIME_API_ATTRS_H_

/*
 * RT_EXT_API_GROUP_BEGIN/END pair is placed around definitions
 * of functions exported by Flang runtime library. They are the entry
 * points that are referenced in the Flang generated code.
 * The macros may be expanded into any construct that is valid to appear
 * at C++ module scope.
 */
#ifndef RT_EXT_API_GROUP_BEGIN
#if defined(OMP_NOHOST_BUILD)
#define RT_EXT_API_GROUP_BEGIN \
  _Pragma("omp begin declare target device_type(nohost)")
#elif defined(OMP_OFFLOAD_BUILD)
#define RT_EXT_API_GROUP_BEGIN _Pragma("omp declare target")
#else
#define RT_EXT_API_GROUP_BEGIN
#endif
#endif /* !defined(RT_EXT_API_GROUP_BEGIN) */

#ifndef RT_EXT_API_GROUP_END
#if defined(OMP_NOHOST_BUILD) || defined(OMP_OFFLOAD_BUILD)
#define RT_EXT_API_GROUP_END _Pragma("omp end declare target")
#else
#define RT_EXT_API_GROUP_END
#endif
#endif /* !defined(RT_EXT_API_GROUP_END) */

/*
 * RT_OFFLOAD_API_GROUP_BEGIN/END pair is placed around definitions
 * of functions that can be referenced in other modules of Flang
 * runtime. For OpenMP offload these functions are made "declare target"
 * making sure they are compiled for the target even though direct
 * references to them from other "declare target" functions may not
 * be seen. Host-only functions should not be put in between these
 * two macros.
 */
#define RT_OFFLOAD_API_GROUP_BEGIN RT_EXT_API_GROUP_BEGIN
#define RT_OFFLOAD_API_GROUP_END RT_EXT_API_GROUP_END

/*
 * RT_VAR_GROUP_BEGIN/END pair is placed around definitions
 * of module scope variables referenced by Flang runtime (directly
 * or indirectly).
 * The macros may be expanded into any construct that is valid to appear
 * at C++ module scope.
 */
#ifndef RT_VAR_GROUP_BEGIN
#define RT_VAR_GROUP_BEGIN RT_EXT_API_GROUP_BEGIN
#endif /* !defined(RT_VAR_GROUP_BEGIN) */

#ifndef RT_VAR_GROUP_END
#define RT_VAR_GROUP_END RT_EXT_API_GROUP_END
#endif /* !defined(RT_VAR_GROUP_END) */

/*
 * Each non-exported function used by Flang runtime (e.g. via
 * calling it or taking its address, etc.) is marked with
 * RT_API_ATTRS. The macros is placed at both declaration and
 * definition of such a function.
 * The macros may be expanded into a construct that is valid
 * to appear as part of a C++ decl-specifier.
 */
#ifndef RT_API_ATTRS
#if defined(__CUDACC__) || defined(__CUDA__)
#define RT_API_ATTRS __host__ __device__
#else
#define RT_API_ATTRS
#endif
#endif /* !defined(RT_API_ATTRS) */

/*
 * Each const/constexpr module scope variable referenced by Flang runtime
 * (directly or indirectly) is marked with RT_CONST_VAR_ATTRS.
 * The macros is placed at both declaration and definition of such a variable.
 * The macros may be expanded into a construct that is valid
 * to appear as part of a C++ decl-specifier.
 */
#ifndef RT_CONST_VAR_ATTRS
#if defined(__CUDACC__) || defined(__CUDA__)
#define RT_CONST_VAR_ATTRS __constant__
#else
#define RT_CONST_VAR_ATTRS
#endif
#endif /* !defined(RT_CONST_VAR_ATTRS) */

/*
 * RT_DEVICE_COMPILATION is defined for any device compilation.
 * Note that it can only be used reliably with compilers that perform
 * separate host and device compilations.
 */
#if ((defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)) || \
    (defined(_OPENMP) && (defined(__AMDGCN__) || defined(__NVPTX__)))
#define RT_DEVICE_COMPILATION 1
#else
#undef RT_DEVICE_COMPILATION
#endif

#endif /* !FORTRAN_RUNTIME_API_ATTRS_H_ */
