/*===---- spirv_builtin_vars.h - SPIR-V built-in ---------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __SPIRV_BUILTIN_VARS_H
#define __SPIRV_BUILTIN_VARS_H

#if __cplusplus >= 201103L
#define __SPIRV_NOEXCEPT noexcept
#else
#define __SPIRV_NOEXCEPT
#endif

#pragma push_macro("__size_t")
#pragma push_macro("__uint32_t")
#pragma push_macro("__uint64_t")
#define __size_t __SIZE_TYPE__
#define __uint32_t __UINT32_TYPE__

#define __SPIRV_overloadable __attribute__((overloadable))
#define __SPIRV_convergent __attribute__((convergent))
#define __SPIRV_inline __attribute__((always_inline))

#ifdef __SYCL_DEVICE_ONLY__
#define __GLOBALAS [[clang::sycl_global]]
#define __LOCALAS [[clang::sycl_local]]
#define __PRIVATEAS [[clang::sycl_private]]
#define __GENERICAS [[clang::sycl_generic]]
#else
#define __GLOBALAS __attribute__((opencl_global))
#define __LOCALAS __attribute__((opencl_local))
#define __PRIVATEAS __attribute__((opencl_private))
#define __GENERICAS __attribute__((opencl_generic))
#endif

// Check if SPIR-V builtins are supported.
// As the translator doesn't use the LLVM intrinsics (which would be emitted if
// we use the SPIR-V builtins) we can't rely on the SPIRV32/SPIRV64 etc macros
// to establish if we can use the builtin alias. We disable builtin altogether
// if we do not intent to use the backend. So instead of use target macros, rely
// on a __has_builtin test.
#if (__has_builtin(__builtin_spirv_num_workgroups))
#define __SPIRV_BUILTIN_ALIAS(builtin)                                         \
  __attribute__((clang_builtin_alias(builtin)))
#else
#define __SPIRV_BUILTIN_ALIAS(builtin)
#endif

// Builtin IDs and sizes

extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_num_workgroups) __size_t
    __spirv_BuiltInNumWorkgroups(int);
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_workgroup_size) __size_t
    __spirv_BuiltInWorkgroupSize(int);
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_workgroup_id) __size_t
    __spirv_BuiltInWorkgroupId(int);
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_local_invocation_id) __size_t
    __spirv_BuiltInLocalInvocationId(int);
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_global_invocation_id) __size_t
    __spirv_BuiltInGlobalInvocationId(int);

extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_global_size) __size_t
    __spirv_BuiltInGlobalSize(int);
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_global_offset) __size_t
    __spirv_BuiltInGlobalOffset(int);
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_subgroup_size) __uint32_t
    __spirv_BuiltInSubgroupSize();
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_subgroup_max_size) __uint32_t
    __spirv_BuiltInSubgroupMaxSize();
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_num_subgroups) __uint32_t
    __spirv_BuiltInNumSubgroups();
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_subgroup_id) __uint32_t
    __spirv_BuiltInSubgroupId();
extern __SPIRV_BUILTIN_ALIAS(__builtin_spirv_subgroup_local_invocation_id)
    __uint32_t __spirv_BuiltInSubgroupLocalInvocationId();

// OpGenericCastToPtrExplicit

extern __SPIRV_overloadable __SPIRV_BUILTIN_ALIAS(
    __builtin_spirv_generic_cast_to_ptr_explicit) void __GLOBALAS
    *__spirv_GenericCastToPtrExplicit_ToGlobal(void __GENERICAS *,
                                               int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable
    __SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit) const
    void __GLOBALAS *__spirv_GenericCastToPtrExplicit_ToGlobal(
        const void __GENERICAS *, int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable __SPIRV_BUILTIN_ALIAS(
    __builtin_spirv_generic_cast_to_ptr_explicit) volatile void __GLOBALAS
    *__spirv_GenericCastToPtrExplicit_ToGlobal(volatile void __GENERICAS *,
                                               int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable
    __SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit) const
    volatile void __GLOBALAS *__spirv_GenericCastToPtrExplicit_ToGlobal(
        const volatile void __GENERICAS *, int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable __SPIRV_BUILTIN_ALIAS(
    __builtin_spirv_generic_cast_to_ptr_explicit) void __LOCALAS
    *__spirv_GenericCastToPtrExplicit_ToLocal(void __GENERICAS *,
                                              int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable
    __SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit) const
    void __LOCALAS *__spirv_GenericCastToPtrExplicit_ToLocal(
        const void __GENERICAS *, int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable __SPIRV_BUILTIN_ALIAS(
    __builtin_spirv_generic_cast_to_ptr_explicit) volatile void __LOCALAS
    *__spirv_GenericCastToPtrExplicit_ToLocal(volatile void __GENERICAS *,
                                              int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable
    __SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit) const
    volatile void __LOCALAS *__spirv_GenericCastToPtrExplicit_ToLocal(
        const volatile void __GENERICAS *, int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable __SPIRV_BUILTIN_ALIAS(
    __builtin_spirv_generic_cast_to_ptr_explicit) void __PRIVATEAS
    *__spirv_GenericCastToPtrExplicit_ToPrivate(void __GENERICAS *,
                                                int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable
    __SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit) const
    void __PRIVATEAS *__spirv_GenericCastToPtrExplicit_ToPrivate(
        const void __GENERICAS *, int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable __SPIRV_BUILTIN_ALIAS(
    __builtin_spirv_generic_cast_to_ptr_explicit) volatile void __PRIVATEAS
    *__spirv_GenericCastToPtrExplicit_ToPrivate(volatile void __GENERICAS *,
                                                int) __SPIRV_NOEXCEPT;
extern __SPIRV_overloadable
    __SPIRV_BUILTIN_ALIAS(__builtin_spirv_generic_cast_to_ptr_explicit) const
    volatile void __PRIVATEAS *__spirv_GenericCastToPtrExplicit_ToPrivate(
        const volatile void __GENERICAS *, int) __SPIRV_NOEXCEPT;

// OpGenericCastToPtr

static __SPIRV_overloadable __SPIRV_inline void __GLOBALAS *
__spirv_GenericCastToPtr_ToGlobal(void __GENERICAS *p, int) __SPIRV_NOEXCEPT {
  return (void __GLOBALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline const void __GLOBALAS *
__spirv_GenericCastToPtr_ToGlobal(const void __GENERICAS *p,
                                  int) __SPIRV_NOEXCEPT {
  return (const void __GLOBALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline volatile void __GLOBALAS *
__spirv_GenericCastToPtr_ToGlobal(volatile void __GENERICAS *p,
                                  int) __SPIRV_NOEXCEPT {
  return (volatile void __GLOBALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline const volatile void __GLOBALAS *
__spirv_GenericCastToPtr_ToGlobal(const volatile void __GENERICAS *p,
                                  int) __SPIRV_NOEXCEPT {
  return (const volatile void __GLOBALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline void __LOCALAS *
__spirv_GenericCastToPtr_ToLocal(void __GENERICAS *p, int) __SPIRV_NOEXCEPT {
  return (void __LOCALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline const void __LOCALAS *
__spirv_GenericCastToPtr_ToLocal(const void __GENERICAS *p,
                                 int) __SPIRV_NOEXCEPT {
  return (const void __LOCALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline volatile void __LOCALAS *
__spirv_GenericCastToPtr_ToLocal(volatile void __GENERICAS *p,
                                 int) __SPIRV_NOEXCEPT {
  return (volatile void __LOCALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline const volatile void __LOCALAS *
__spirv_GenericCastToPtr_ToLocal(const volatile void __GENERICAS *p,
                                 int) __SPIRV_NOEXCEPT {
  return (const volatile void __LOCALAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline void __PRIVATEAS *
__spirv_GenericCastToPtr_ToPrivate(void __GENERICAS *p, int) __SPIRV_NOEXCEPT {
  return (void __PRIVATEAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline const void __PRIVATEAS *
__spirv_GenericCastToPtr_ToPrivate(const void __GENERICAS *p,
                                   int) __SPIRV_NOEXCEPT {
  return (const void __PRIVATEAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline volatile void __PRIVATEAS *
__spirv_GenericCastToPtr_ToPrivate(volatile void __GENERICAS *p,
                                   int) __SPIRV_NOEXCEPT {
  return (volatile void __PRIVATEAS *)p;
}
static __SPIRV_overloadable __SPIRV_inline const volatile void __PRIVATEAS *
__spirv_GenericCastToPtr_ToPrivate(const volatile void __GENERICAS *p,
                                   int) __SPIRV_NOEXCEPT {
  return (const volatile void __PRIVATEAS *)p;
}

#pragma pop_macro("__size_t")
#pragma pop_macro("__uint32_t")
#pragma pop_macro("__uint64_t")

#undef __SPIRV_overloadable
#undef __SPIRV_convergent
#undef __SPIRV_inline

#undef __GLOBALAS
#undef __LOCALAS
#undef __PRIVATEAS
#undef __GENERICAS

#undef __SPIRV_BUILTIN_ALIAS
#undef __SPIRV_NOEXCEPT

#endif /* __SPIRV_BUILTIN_VARS_H */
