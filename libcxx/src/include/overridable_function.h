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
// This is a low-level utility which does not work on all platforms, since it needs to
// make assumptions about the object file format in use. This currently works with Mach-O
// files (used on Darwin) and with ELF files (used on Linux and others). On platforms
// where we know how to implement this detection, the macro
// _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION  is defined to 1, and it is defined to 0 on
// other platforms.
//
// How does this work?
// -------------------
//
// Let's say we want to check whether a weak function `f` has been overridden by the user.
// The general mechanism works by defining a local symbol `__impl_ref<f>::__impl_` with
// the same address as `f` as a constant expression using direct PC-relative
// materialization thus pointing at the symbol defined in the same TU. At runtime, it
// compares the address of `__impl_ref<f>::__impl_` with the address of `f` loaded from
// GOT: if `f` was overridden by the user in another TU, the addresses will be different.
//
// Important note
// --------------
//
// This mechanism should never be used outside of the libc++ built library. In particular,
// attempting to use this within the libc++ headers will not work at all because we don't
// want to be defining special sections inside user's executables which use our headers.
//

#if defined(_LIBCPP_OBJECT_FORMAT_MACHO) || (defined(_LIBCPP_OBJECT_FORMAT_ELF) && !defined(__NVPTX__))

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 1

_LIBCPP_BEGIN_NAMESPACE_STD

namespace {

// This is used to prevent TBAA from optimizing away the function pointer comparison.
template <typename T>
[[nodiscard]] inline _LIBCPP_HIDE_FROM_ABI T* __launder_function_pointer(T* __ptr) noexcept {
  __asm__ volatile("" : "+r"(__ptr));
  return __ptr;
}

} // namespace

template <auto* _Func>
struct __impl_ref;

// __impl_ref<...>::__impl_ is expected to be defined elsewhere, so the compiler emits
// assembly references to the mangled symbol with no definition. This template saves us
// the trouble of providing manual declarations for overloads with some other local name
// for each function name being overloaded (operator new, operator new[], etc.).
template <typename _Ret, typename... _Args, _Ret (*_Func)(_Args...)>
struct __impl_ref<_Func> {
  [[gnu::visibility("hidden")]] static _Ret __impl_(_Args...);
};

// This takes a function type template argument first so that the second non-type template
// argument (pointer to the public function) gets the benefit of type-aware overload
// resolution, rather than having to use a static_cast.
template <typename T, T* _Func>
_LIBCPP_HIDE_FROM_ABI inline bool __is_function_overridden() noexcept {
#  if !defined(_LIBCPP_CLANG_VER) || _LIBCPP_CLANG_VER >= 2101
  __asm__("%cc0 = %cc1" : : "X"(__impl_ref<_Func>::__impl_), "X"(_Func));
#  else
  __asm__("%c0 = %c1" : : "X"(__impl_ref<_Func>::__impl_), "X"(_Func));
#  endif
  // This just has the compiler compare the two symbols. For PIC mode, this will do a
  // direct PC-relative materialization for __impl_ref<...>::__impl_ and a GOT load for
  // the _Func symbol. The compiler thinks __impl_ref<...>::__impl_ is defined elsewhere
  // at link time and will be an undefined symbol. It doesn't know that the __asm__ tells
  // the assembler to define it as a local symbol.
  return __launder_function_pointer(_Func) != __impl_ref<_Func>::__impl_;
}

_LIBCPP_END_NAMESPACE_STD

#else

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 0

#endif

#endif // _LIBCPP_SRC_INCLUDE_OVERRIDABLE_FUNCTION_H
