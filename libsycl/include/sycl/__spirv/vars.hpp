//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains SPIRV builtin helpers needed for kernel invocations
/// (parallel_for).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___SPIRV_VARS_HPP
#define _LIBSYCL___SPIRV_VARS_HPP

#ifdef __SYCL_DEVICE_ONLY__
#  include <__clang_spirv_builtins.h>
#endif

#include <cstddef>
#include <cstdint>

namespace __spirv {

// The __builtin_spirv_* aliases exist for SPIR-V targets only. An
// external-linkage kernel body is still emitted for the host and must link, so
// the host branch returns values that are never read. Keep the include guarded.
#ifdef __SYCL_DEVICE_ONLY__
#  define __SPIRV_INDEXED_BUILTIN(POSTFIX, ID) __spirv_##POSTFIX(ID)
#  define __SPIRV_SCALAR_BUILTIN(POSTFIX) __spirv_##POSTFIX()
#else
#  define __SPIRV_INDEXED_BUILTIN(POSTFIX, ID) 0
#  define __SPIRV_SCALAR_BUILTIN(POSTFIX) 0
#endif

// Helper function templates to initialize and get vector component from SPIR-V
// built-in variables
#define __SPIRV_DEFINE_INIT_AND_GET_HELPERS(POSTFIX)                           \
  template <int ID> std::size_t get##POSTFIX();                                \
  template <> inline std::size_t get##POSTFIX<0>() {                           \
    return __SPIRV_INDEXED_BUILTIN(POSTFIX, 0);                                \
  }                                                                            \
  template <> inline std::size_t get##POSTFIX<1>() {                           \
    return __SPIRV_INDEXED_BUILTIN(POSTFIX, 1);                                \
  }                                                                            \
  template <> inline std::size_t get##POSTFIX<2>() {                           \
    return __SPIRV_INDEXED_BUILTIN(POSTFIX, 2);                                \
  }                                                                            \
                                                                               \
  template <int Dim, class DstT> struct InitSizesST##POSTFIX;                  \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<1, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<0>()}; }                     \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<2, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<1>(), get##POSTFIX<0>()}; }  \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<3, DstT> {                 \
    static DstT initSize() {                                                   \
      return {get##POSTFIX<2>(), get##POSTFIX<1>(), get##POSTFIX<0>()};        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <int Dims, class DstT> DstT init##POSTFIX() {                       \
    return InitSizesST##POSTFIX<Dims, DstT>::initSize();                       \
  }

__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInGlobalSize)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInGlobalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInGlobalOffset)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInWorkgroupId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInLocalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInWorkgroupSize)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInNumWorkgroups)

#undef __SPIRV_DEFINE_INIT_AND_GET_HELPERS

// Helper functions to get scalar SPIR-V built-in variables
#define __SPIRV_DEFINE_GET_HELPER(POSTFIX)                                     \
  inline std::uint32_t get##POSTFIX() {                                        \
    return __SPIRV_SCALAR_BUILTIN(POSTFIX);                                    \
  }

__SPIRV_DEFINE_GET_HELPER(BuiltInSubgroupId)
__SPIRV_DEFINE_GET_HELPER(BuiltInSubgroupLocalInvocationId)
__SPIRV_DEFINE_GET_HELPER(BuiltInSubgroupSize)
__SPIRV_DEFINE_GET_HELPER(BuiltInSubgroupMaxSize)
__SPIRV_DEFINE_GET_HELPER(BuiltInNumSubgroups)

#undef __SPIRV_DEFINE_GET_HELPER

#undef __SPIRV_INDEXED_BUILTIN
#undef __SPIRV_SCALAR_BUILTIN

} // namespace __spirv

#endif // _LIBSYCL___SPIRV_VARS_HPP
