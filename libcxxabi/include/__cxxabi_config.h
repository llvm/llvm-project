//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ____CXXABI_CONFIG_H
#define ____CXXABI_CONFIG_H

#if defined(__arm__) && !defined(__USING_SJLJ_EXCEPTIONS__) &&                 \
    !defined(__ARM_DWARF_EH__) && !defined(__SEH__)
#define _LIBCXXABI_ARM_EHABI
#endif

#if !defined(__has_attribute)
#define __has_attribute(_attribute_) 0
#endif

#if defined(__clang__)
#  define _LIBCXXABI_COMPILER_CLANG
#  ifndef __apple_build_version__
#    define _LIBCXXABI_CLANG_VER (__clang_major__ * 100 + __clang_minor__)
#  endif
#elif defined(__GNUC__)
#  define _LIBCXXABI_COMPILER_GCC
#elif defined(_MSC_VER)
#  define _LIBCXXABI_COMPILER_MSVC
#elif defined(__IBMCPP__)
#  define _LIBCXXABI_COMPILER_IBM
#endif

#if defined(_WIN32)
 #if defined(_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS) || (defined(__MINGW32__) && !defined(_LIBCXXABI_BUILDING_LIBRARY))
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS
  #define _LIBCXXABI_FUNC_VIS
  #define _LIBCXXABI_TYPE_VIS
 #elif defined(_LIBCXXABI_BUILDING_LIBRARY)
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS __declspec(dllexport)
  #define _LIBCXXABI_FUNC_VIS __declspec(dllexport)
  #define _LIBCXXABI_TYPE_VIS __declspec(dllexport)
 #else
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS __declspec(dllimport)
  #define _LIBCXXABI_FUNC_VIS __declspec(dllimport)
  #define _LIBCXXABI_TYPE_VIS __declspec(dllimport)
 #endif
#else
 #if !defined(_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS)
  #define _LIBCXXABI_HIDDEN __attribute__((__visibility__("hidden")))
  #define _LIBCXXABI_DATA_VIS __attribute__((__visibility__("default")))
  #define _LIBCXXABI_FUNC_VIS __attribute__((__visibility__("default")))
  #if __has_attribute(__type_visibility__)
   #define _LIBCXXABI_TYPE_VIS __attribute__((__type_visibility__("default")))
  #else
   #define _LIBCXXABI_TYPE_VIS __attribute__((__visibility__("default")))
  #endif
 #else
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS
  #define _LIBCXXABI_FUNC_VIS
  #define _LIBCXXABI_TYPE_VIS
 #endif
#endif

#if defined(_LIBCXXABI_COMPILER_MSVC)
#define _LIBCXXABI_WEAK
#else
#define _LIBCXXABI_WEAK __attribute__((__weak__))
#endif

#if defined(__clang__)
#define _LIBCXXABI_COMPILER_CLANG
#elif defined(__GNUC__)
#define _LIBCXXABI_COMPILER_GCC
#endif

#if __has_attribute(__no_sanitize__) && defined(_LIBCXXABI_COMPILER_CLANG)
#define _LIBCXXABI_NO_CFI __attribute__((__no_sanitize__("cfi")))
#else
#define _LIBCXXABI_NO_CFI
#endif

// wasm32 follows the arm32 ABI convention of using 32-bit guard.
#if defined(__arm__) || defined(__wasm32__) || defined(__ARM64_ARCH_8_32__)
#  define _LIBCXXABI_GUARD_ABI_ARM
#endif

#if defined(_LIBCXXABI_COMPILER_CLANG)
#  if !__has_feature(cxx_exceptions)
#    define _LIBCXXABI_NO_EXCEPTIONS
#  endif
#elif defined(_LIBCXXABI_COMPILER_GCC) && !defined(__EXCEPTIONS)
#  define _LIBCXXABI_NO_EXCEPTIONS
#endif

#if defined(_WIN32)
#define _LIBCXXABI_DTOR_FUNC __thiscall
#else
#define _LIBCXXABI_DTOR_FUNC
#endif

#if __has_include(<ptrauth.h>)
#  include <ptrauth.h>
#endif

#if __has_feature(ptrauth_calls)

// ptrauth_string_discriminator("__cxa_exception::actionRecord") == 0xFC91
#  define __ptrauth_cxxabi_action_record __ptrauth(ptrauth_key_process_dependent_data, 1, 0xFC91)

// ptrauth_string_discriminator("__cxa_exception::languageSpecificData") == 0xE8EE
#  define __ptrauth_cxxabi_lsd __ptrauth(ptrauth_key_process_dependent_data, 1, 0xE8EE)

// ptrauth_string_discriminator("__cxa_exception::catchTemp") == 0xFA58
#  define __ptrauth_cxxabi_catch_temp_disc 0xFA58
#  define __ptrauth_cxxabi_catch_temp_key ptrauth_key_process_dependent_data
#  define __ptrauth_cxxabi_catch_temp __ptrauth(__ptrauth_cxxabi_catch_temp_key, 1, __ptrauth_cxxabi_catch_temp_disc)

// ptrauth_string_discriminator("__cxa_exception::adjustedPtr") == 0x99E4
#  define __ptrauth_cxxabi_adjusted_ptr __ptrauth(ptrauth_key_process_dependent_data, 1, 0x99E4)

// ptrauth_string_discriminator("__cxa_exception::unexpectedHandler") == 0x99A9
#  define __ptrauth_cxxabi_unexpected_handler __ptrauth(ptrauth_key_function_pointer, 1, 0x99A9)

// ptrauth_string_discriminator("__cxa_exception::terminateHandler") == 0x0886)
#  define __ptrauth_cxxabi_terminate_handler __ptrauth(ptrauth_key_function_pointer, 1, 0x886)

// ptrauth_string_discriminator("__cxa_exception::exceptionDestructor") == 0xC088
#  define __ptrauth_cxxabi_exception_destructor __ptrauth(ptrauth_key_function_pointer, 1, 0xC088)

#else

#  define __ptrauth_cxxabi_action_record
#  define __ptrauth_cxxabi_lsd
#  define __ptrauth_cxxabi_catch_temp
#  define __ptrauth_cxxabi_adjusted_ptr
#  define __ptrauth_cxxabi_unexpected_handler
#  define __ptrauth_cxxabi_terminate_handler
#  define __ptrauth_cxxabi_exception_destructor

#endif

#if __cplusplus < 201103L
#  define _LIBCXXABI_NOEXCEPT throw()
#else
#  define _LIBCXXABI_NOEXCEPT noexcept
#endif

#endif // ____CXXABI_CONFIG_H
