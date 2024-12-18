// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SRC_INCLUDE_OVERRIDABLE_FUNCTION_H
#define _LIBCPP_SRC_INCLUDE_OVERRIDABLE_FUNCTION_H

#include <__config>
#include <cstdint>

#if __has_feature(ptrauth_calls)
#  include <ptrauth.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

//
// This file provides the std::__is_function_overridden utility, which allows checking
// whether an overridable function (typically a weak symbol) like `operator new`
// has been overridden by a user or not.
//
// This is a low-level utility which does not work on all platforms, since it needs
// to make assumptions about the object file format in use. Furthermore, it requires
// the "base definition" of the function (the one we want to check whether it has been
// overridden) to be defined using the _LIBCPP_OVERRIDABLE_FUNCTION macro.
//
// This currently works with Mach-O files (used on Darwin) and with ELF files (used on Linux
// and others). On platforms where we know how to implement this detection, the macro
// _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION is defined to 1, and it is defined to 0 on
// other platforms. The _LIBCPP_OVERRIDABLE_FUNCTION macro expands to regular function
// definition on unsupported platforms so that it can be used to decorate functions
// regardless of whether detection is actually supported.
//
// How does this work?
// -------------------
//
// Let's say we want to check whether a weak function `f` has been overridden by the user.
// The general mechanism works by defining a symbol `f_impl__` and a weak alias `f` via the
// _LIBCPP_OVERRIDABLE_FUNCTION macro.
//
// Then, when comes the time to check whether the function has been overridden, we take
// the address of the function `f` and we check whether it is different from `f_impl__`.
// If so it means the function was overriden by the user.
//
// Important note
// --------------
//
// This mechanism should never be used outside of the libc++ built library. Functions defined
// with this macro must be defined at global scope.
//

#if defined(_LIBCPP_OBJECT_FORMAT_MACHO)

_LIBCPP_BEGIN_NAMESPACE_STD

template <auto _Func>
_LIBCPP_HIDE_FROM_ABI constexpr bool __is_function_overridden();

_LIBCPP_END_NAMESPACE_STD

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 1
#  define _LIBCPP_OVERRIDABLE_FUNCTION(symbol, type, name, arglist)                                                    \
    static type symbol##_impl__ arglist __asm__("_" _LIBCPP_TOSTRING(symbol));                                         \
    __asm__(".globl _" _LIBCPP_TOSTRING(symbol));                                                                      \
    __asm__(".weak_definition _" _LIBCPP_TOSTRING(symbol));                                                            \
    extern __typeof(symbol##_impl__) name __attribute__((weak_import));                                                \
    _LIBCPP_BEGIN_NAMESPACE_STD                                                                                        \
    template <>                                                                                                        \
    bool __is_function_overridden<static_cast<type(*) arglist>(name)>() {                                              \
      return static_cast<type(*) arglist>(name) != symbol##_impl__;                                                    \
    }                                                                                                                  \
    _LIBCPP_END_NAMESPACE_STD                                                                                          \
    static type symbol##_impl__ arglist

#elif defined(_LIBCPP_OBJECT_FORMAT_ELF)

_LIBCPP_BEGIN_NAMESPACE_STD

template <auto _Func>
_LIBCPP_HIDE_FROM_ABI constexpr bool __is_function_overridden();

_LIBCPP_END_NAMESPACE_STD

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 1
#  define _LIBCPP_OVERRIDABLE_FUNCTION(symbol, type, name, arglist)                                                    \
    static type symbol##_impl__ arglist __asm__(_LIBCPP_TOSTRING(symbol##_impl__));                                    \
    [[gnu::weak, gnu::alias(_LIBCPP_TOSTRING(symbol##_impl__))]] type name arglist;                                    \
    _LIBCPP_BEGIN_NAMESPACE_STD                                                                                        \
    template <>                                                                                                        \
    bool __is_function_overridden<static_cast<type(*) arglist>(name)>() {                                              \
      return static_cast<type(*) arglist>(name) != symbol##_impl__;                                                    \
    }                                                                                                                  \
    _LIBCPP_END_NAMESPACE_STD                                                                                          \
    static type symbol##_impl__ arglist

#else

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 0
#  define _LIBCPP_OVERRIDABLE_FUNCTION(symbol, type, name, arglist) _LIBCPP_WEAK type name arglist

#endif

#endif // _LIBCPP_SRC_INCLUDE_OVERRIDABLE_FUNCTION_H
