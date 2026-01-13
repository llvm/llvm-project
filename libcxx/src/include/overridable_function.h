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
// overridden) to be defined using the OVERRIDABLE_FUNCTION macro.
//
// This currently works with Mach-O files (used on Darwin) and with ELF files (used on Linux
// and others). On platforms where we know how to implement this detection, the macro
// _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION is defined to 1, and it is defined to 0 on
// other platforms. The OVERRIDABLE_FUNCTION macro is defined to perform a normal
// function definition on unsupported platforms so that it can be used to define functions
// regardless of whether detection is actually supported.
//
// Important note
// --------------
//
// This mechanism should never be used outside of the libc++ built library. In particular,
// attempting to use this within the libc++ headers will not work at all because we don't
// want to be defining special sections inside user's executables which use our headers.
//

#if defined(_LIBCPP_OBJECT_FORMAT_MACHO) || (defined(_LIBCPP_OBJECT_FORMAT_ELF) && !defined(__NVPTX__))

// Template type can be partially specialized to dissect argument type, unlike
// a template function.
template <auto* _Func>
struct ImplRef;

// Partial specialization can tease out the components of the function type.
// ImplRef<...>::Impl is expected to be defined elsewhere, so the compiler will
// just emit assembly references to the mangled symbol with no definition.
// This template is just saving us the trouble of doing separate manual
// declarations for overload with some other local name for each function name
// being overloaded (operator new, operator new[], etc.).
template <typename _Ret, typename... _Args, _Ret (*_Func)(_Args...)>
struct ImplRef<_Func> {
  [[gnu::visibility("hidden")]] static _Ret Impl(_Args...);
};

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 1
#  define OVERRIDABLE_FUNCTION [[gnu::weak]]

_LIBCPP_BEGIN_NAMESPACE_STD
template <typename T, T* _Func>
_LIBCPP_HIDE_FROM_ABI inline bool __is_function_overridden() noexcept {
  // This just has the compiler compare the two symbols.  For PIC mode, this will
  // do a direct PC-relative materialization for ImplRef<...>::Impl and a GOT
  // load for the _Func symbol.  The compiler thinks ImplRef<...>::Impl is
  // defined elsewhere at link time and will be an undefined symbol.  It doesn't
  // know that the __asm__ tells the assembler to define it as a local symbol.
#  if !defined(_LIBCPP_CLANG_VER) || _LIBCPP_CLANG_VER >= 2101
  __asm__("%cc0 = %cc1" : : "X"(ImplRef<_Func>::Impl), "X"(_Func));
#  else
  __asm__("%c0 = %c1" : : "X"(ImplRef<_Func>::Impl), "X"(_Func));
#  endif
  return _Func == ImplRef<_Func>::Impl;
}
_LIBCPP_END_NAMESPACE_STD

#else

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 0
#  define OVERRIDABLE_FUNCTION [[gnu::weak]]

#endif

#endif // _LIBCPP_SRC_INCLUDE_OVERRIDABLE_FUNCTION_H
