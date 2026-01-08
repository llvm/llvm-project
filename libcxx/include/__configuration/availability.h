// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_AVAILABILITY_H
#define _LIBCPP___CONFIGURATION_AVAILABILITY_H

#include <__configuration/compiler.h>
#include <__configuration/language.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// This file defines a framework that can be used by vendors to encode the version of an operating system that various
// features of libc++ has been shipped in. This is primarily intended to allow safely deploying an executable built with
// a new version of the library on a platform containing an older version of the built library.
// Detailed documentation for this can be found at https://libcxx.llvm.org/VendorDocumentation.html#availability-markup

// Availability markup is disabled when building the library, or when a non-Clang
// compiler is used because only Clang supports the necessary attributes.
//
// We also allow users to force-disable availability markup via the `_LIBCPP_DISABLE_AVAILABILITY`
// macro because that is the only way to work around a Clang bug related to availability
// attributes: https://llvm.org/PR134151.
// Once that bug has been fixed, we should remove the macro.
#if defined(_LIBCPP_BUILDING_LIBRARY) || defined(_LIBCXXABI_BUILDING_LIBRARY) ||                                       \
    !defined(_LIBCPP_COMPILER_CLANG_BASED) || defined(_LIBCPP_DISABLE_AVAILABILITY)
#  undef _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS
#  define _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS 0
#endif

// When availability annotations are disabled, we take for granted that features introduced
// in all versions of the library are available.
#if !_LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS

#  define _LIBCPP_INTRODUCED_IN_LLVM_22 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_22_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_21 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_21_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_20 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_20_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_19 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_19_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_18 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_18_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_16 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_16_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_15 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_15_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_14 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_14_ATTRIBUTE /* nothing */

#  define _LIBCPP_INTRODUCED_IN_LLVM_12 1
#  define _LIBCPP_INTRODUCED_IN_LLVM_12_ATTRIBUTE /* nothing */

#elif defined(__APPLE__)

// clang-format off

// LLVM 22
// TODO: Fill this in
#  define _LIBCPP_INTRODUCED_IN_LLVM_22 0
#  define _LIBCPP_INTRODUCED_IN_LLVM_22_ATTRIBUTE __attribute__((unavailable))

// LLVM 21
// TODO: Fill this in
#  define _LIBCPP_INTRODUCED_IN_LLVM_21 0
#  define _LIBCPP_INTRODUCED_IN_LLVM_21_ATTRIBUTE __attribute__((unavailable))

// LLVM 20
//
// Note that versions for most Apple OSes were bumped forward and aligned in that release.
#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 260000) ||       \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 260000) ||     \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 260000) ||             \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 260000) ||       \
      (defined(__ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__ < 100000)
#    define _LIBCPP_INTRODUCED_IN_LLVM_20 0
#  else
#    define _LIBCPP_INTRODUCED_IN_LLVM_20 1
#  endif
#  define _LIBCPP_INTRODUCED_IN_LLVM_20_ATTRIBUTE                                                                 \
    __attribute__((availability(macos, strict, introduced = 26.0)))                                               \
    __attribute__((availability(ios, strict, introduced = 26.0)))                                                 \
    __attribute__((availability(tvos, strict, introduced = 26.0)))                                                \
    __attribute__((availability(watchos, strict, introduced = 26.0)))                                             \
    __attribute__((availability(bridgeos, strict, introduced = 10.0)))

// LLVM 19
#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 150400) ||       \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 180400) ||     \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 180400) ||             \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 110400) ||       \
      (defined(__ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__ < 90400)
#    define _LIBCPP_INTRODUCED_IN_LLVM_19 0
#  else
#    define _LIBCPP_INTRODUCED_IN_LLVM_19 1
#  endif
#  define _LIBCPP_INTRODUCED_IN_LLVM_19_ATTRIBUTE                                                                 \
    __attribute__((availability(macos, strict, introduced = 15.4)))                                               \
    __attribute__((availability(ios, strict, introduced = 18.4)))                                                 \
    __attribute__((availability(tvos, strict, introduced = 18.4)))                                                \
    __attribute__((availability(watchos, strict, introduced = 11.4)))                                             \
    __attribute__((availability(bridgeos, strict, introduced = 9.4)))

// LLVM 18
#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 150000) ||       \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 180000) ||     \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 180000) ||             \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 110000) ||       \
      (defined(__ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__ < 90000) ||      \
      (defined(__ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__ < 240000)
#    define _LIBCPP_INTRODUCED_IN_LLVM_18 0
#  else
#    define _LIBCPP_INTRODUCED_IN_LLVM_18 1
#  endif
#  define _LIBCPP_INTRODUCED_IN_LLVM_18_ATTRIBUTE                                                                 \
    __attribute__((availability(macos, strict, introduced = 15.0)))                                               \
    __attribute__((availability(ios, strict, introduced = 18.0)))                                                 \
    __attribute__((availability(tvos, strict, introduced = 18.0)))                                                \
    __attribute__((availability(watchos, strict, introduced = 11.0)))                                             \
    __attribute__((availability(bridgeos, strict, introduced = 9.0)))                                             \
    __attribute__((availability(driverkit, strict, introduced = 24.0)))

// LLVM 16
#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 140000) ||       \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 170000) ||     \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 170000) ||             \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 100000) ||       \
      (defined(__ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__ < 80000) ||      \
      (defined(__ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__ < 230000)
#    define _LIBCPP_INTRODUCED_IN_LLVM_16 0
#  else
#    define _LIBCPP_INTRODUCED_IN_LLVM_16 1
#  endif
#  define _LIBCPP_INTRODUCED_IN_LLVM_16_ATTRIBUTE                                                                 \
    __attribute__((availability(macos, strict, introduced = 14.0)))                                               \
    __attribute__((availability(ios, strict, introduced = 17.0)))                                                 \
    __attribute__((availability(tvos, strict, introduced = 17.0)))                                                \
    __attribute__((availability(watchos, strict, introduced = 10.0)))                                             \
    __attribute__((availability(bridgeos, strict, introduced = 8.0)))                                             \
    __attribute__((availability(driverkit, strict, introduced = 23.0)))

// LLVM 15
#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 130300) ||   \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 160300) || \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 160300) ||         \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 90300) ||    \
      (defined(__ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__ < 70500) ||  \
      (defined(__ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__ < 220400)
#    define _LIBCPP_INTRODUCED_IN_LLVM_15 0
#  else
#    define _LIBCPP_INTRODUCED_IN_LLVM_15 1
#  endif
#  define _LIBCPP_INTRODUCED_IN_LLVM_15_ATTRIBUTE                                                                 \
    __attribute__((availability(macos, strict, introduced = 13.3)))                                               \
    __attribute__((availability(ios, strict, introduced = 16.3)))                                                 \
    __attribute__((availability(tvos, strict, introduced = 16.3)))                                                \
    __attribute__((availability(watchos, strict, introduced = 9.3)))                                              \
    __attribute__((availability(bridgeos, strict, introduced = 7.5)))                                             \
    __attribute__((availability(driverkit, strict, introduced = 22.4)))

// LLVM 14
#  define _LIBCPP_INTRODUCED_IN_LLVM_14 _LIBCPP_INTRODUCED_IN_LLVM_15
#  define _LIBCPP_INTRODUCED_IN_LLVM_14_ATTRIBUTE _LIBCPP_INTRODUCED_IN_LLVM_15_ATTRIBUTE

// LLVM 12
#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 120300)   ||     \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 150300) ||     \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 150300)         ||     \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 80300)    ||     \
      (defined(__ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_BRIDGE_OS_VERSION_MIN_REQUIRED__ < 60000)  ||     \
      (defined(__ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__ < 210300)
#    define _LIBCPP_INTRODUCED_IN_LLVM_12 0
#  else
#    define _LIBCPP_INTRODUCED_IN_LLVM_12 1
#  endif
#  define _LIBCPP_INTRODUCED_IN_LLVM_12_ATTRIBUTE                                                                 \
    __attribute__((availability(macos, strict, introduced = 12.3)))                                               \
    __attribute__((availability(ios, strict, introduced = 15.3)))                                                 \
    __attribute__((availability(tvos, strict, introduced = 15.3)))                                                \
    __attribute__((availability(watchos, strict, introduced = 8.3)))                                              \
    __attribute__((availability(bridgeos, strict, introduced = 6.0)))                                             \
    __attribute__((availability(driverkit, strict, introduced = 21.3)))

#  if (defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)  && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__  < 101300) || \
      (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 150000) || \
      (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__)     && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__     < 150000) || \
      (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__)  && __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__  < 70000)  || \
      (defined(__ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_DRIVERKIT_VERSION_MIN_REQUIRED__ < 190000)
#  warning "The selected platform is no longer supported by libc++."
#  endif

#else

// ...New vendors can add availability markup here...

#  error                                                                                                               \
      "It looks like you're trying to enable vendor availability markup, but you haven't defined the corresponding macros yet!"

#endif

// This controls the availability of new implementation of std::atomic's
// wait, notify_one and notify_all. The new implementation uses
// the native atomic wait/notify operations on platforms that support them
// based on the size of the atomic type, instead of the type itself.
#define _LIBCPP_AVAILABILITY_HAS_NEW_SYNC _LIBCPP_INTRODUCED_IN_LLVM_22
#define _LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_INTRODUCED_IN_LLVM_22_ATTRIBUTE

// Enable additional explicit instantiations of iostreams components. This
// reduces the number of weak definitions generated in programs that use
// iostreams by providing a single strong definition in the shared library.
//
// TODO: Enable additional explicit instantiations on GCC once it supports exclude_from_explicit_instantiation,
//       or once libc++ doesn't use the attribute anymore.
// TODO: Enable them on Windows once https://llvm.org/PR41018 has been fixed.
#if !defined(_LIBCPP_COMPILER_GCC) && !defined(_WIN32)
#  define _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1 _LIBCPP_INTRODUCED_IN_LLVM_12
#else
#  define _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1 0
#endif

// This controls the availability of floating-point std::to_chars functions.
// These overloads were added later than the integer overloads.
#define _LIBCPP_AVAILABILITY_HAS_TO_CHARS_FLOATING_POINT _LIBCPP_INTRODUCED_IN_LLVM_14
#define _LIBCPP_AVAILABILITY_TO_CHARS_FLOATING_POINT _LIBCPP_INTRODUCED_IN_LLVM_14_ATTRIBUTE

// This controls whether the library claims to provide a default verbose
// termination function, and consequently whether the headers will try
// to use it when the mechanism isn't overriden at compile-time.
#define _LIBCPP_AVAILABILITY_HAS_VERBOSE_ABORT _LIBCPP_INTRODUCED_IN_LLVM_15
#define _LIBCPP_AVAILABILITY_VERBOSE_ABORT _LIBCPP_INTRODUCED_IN_LLVM_15_ATTRIBUTE

// This controls the availability of the C++17 std::pmr library,
// which is implemented in large part in the built library.
//
// TODO: Enable std::pmr markup once https://llvm.org/PR40340 has been fixed
//       Until then, it is possible for folks to try to use `std::pmr` when back-deploying to targets that don't support
//       it and it'll be a load-time error, but we don't have a good alternative because the library won't compile if we
//       use availability annotations until that bug has been fixed.
#define _LIBCPP_AVAILABILITY_HAS_PMR _LIBCPP_INTRODUCED_IN_LLVM_16
#define _LIBCPP_AVAILABILITY_PMR

// These macros controls the availability of __cxa_init_primary_exception
// in the built library, which std::make_exception_ptr might use
// (see libcxx/include/__exception/exception_ptr.h).
#define _LIBCPP_AVAILABILITY_HAS_INIT_PRIMARY_EXCEPTION _LIBCPP_INTRODUCED_IN_LLVM_18
#define _LIBCPP_AVAILABILITY_INIT_PRIMARY_EXCEPTION _LIBCPP_INTRODUCED_IN_LLVM_18_ATTRIBUTE

// This controls the availability of C++23 <print>, which
// has a dependency on the built library (it needs access to
// the underlying buffer types of std::cout, std::cerr, and std::clog.
#define _LIBCPP_AVAILABILITY_HAS_PRINT _LIBCPP_INTRODUCED_IN_LLVM_18
#define _LIBCPP_AVAILABILITY_PRINT _LIBCPP_INTRODUCED_IN_LLVM_18_ATTRIBUTE

// This controls the availability of the C++20 time zone database.
// The parser code is built in the library.
#define _LIBCPP_AVAILABILITY_HAS_TZDB _LIBCPP_INTRODUCED_IN_LLVM_19
#define _LIBCPP_AVAILABILITY_TZDB _LIBCPP_INTRODUCED_IN_LLVM_19_ATTRIBUTE

// These macros determine whether we assume that std::bad_function_call and
// std::bad_expected_access provide a key function in the dylib. This allows
// centralizing their vtable and typeinfo instead of having all TUs provide
// a weak definition that then gets deduplicated.
#define _LIBCPP_AVAILABILITY_HAS_BAD_FUNCTION_CALL_KEY_FUNCTION _LIBCPP_INTRODUCED_IN_LLVM_19
#define _LIBCPP_AVAILABILITY_BAD_FUNCTION_CALL_KEY_FUNCTION _LIBCPP_INTRODUCED_IN_LLVM_19_ATTRIBUTE
#define _LIBCPP_AVAILABILITY_HAS_BAD_EXPECTED_ACCESS_KEY_FUNCTION _LIBCPP_INTRODUCED_IN_LLVM_19
#define _LIBCPP_AVAILABILITY_BAD_EXPECTED_ACCESS_KEY_FUNCTION _LIBCPP_INTRODUCED_IN_LLVM_19_ATTRIBUTE

// This controls the availability of floating-point std::from_chars functions.
// These overloads were added later than the integer overloads.
#define _LIBCPP_AVAILABILITY_HAS_FROM_CHARS_FLOATING_POINT _LIBCPP_INTRODUCED_IN_LLVM_20
#define _LIBCPP_AVAILABILITY_FROM_CHARS_FLOATING_POINT _LIBCPP_INTRODUCED_IN_LLVM_20_ATTRIBUTE

// This controls whether `std::__hash_memory` is available in the dylib, which
// is used for some `std::hash` specializations.
#define _LIBCPP_AVAILABILITY_HAS_HASH_MEMORY _LIBCPP_INTRODUCED_IN_LLVM_21
// No attribute, since we've had hash in the headers before

// This controls whether we provide a message for `bad_function_call::what()` that specific to `std::bad_function_call`.
// See https://wg21.link/LWG2233. This requires `std::bad_function_call::what()` to be available in the dylib.
#define _LIBCPP_AVAILABILITY_HAS_BAD_FUNCTION_CALL_GOOD_WHAT_MESSAGE _LIBCPP_INTRODUCED_IN_LLVM_21
// No attribute, since we've had bad_function_call::what() in the headers before

// Define availability attributes that depend on both
// _LIBCPP_HAS_EXCEPTIONS and _LIBCPP_HAS_RTTI.
#if !_LIBCPP_HAS_EXCEPTIONS || !_LIBCPP_HAS_RTTI
#  undef _LIBCPP_AVAILABILITY_HAS_INIT_PRIMARY_EXCEPTION
#  undef _LIBCPP_AVAILABILITY_INIT_PRIMARY_EXCEPTION
#  define _LIBCPP_AVAILABILITY_HAS_INIT_PRIMARY_EXCEPTION 0
#  define _LIBCPP_AVAILABILITY_INIT_PRIMARY_EXCEPTION
#endif

#endif // _LIBCPP___CONFIGURATION_AVAILABILITY_H
