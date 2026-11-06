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
// When pointer authentication is used, the above mechanism doesn't work (yet) so we use
// a different strategy placing `f`'s definition (in the libc++ built library) inside
// a special section, which we do using the `__section__` attribute via the
// OVERRIDABLE_FUNCTION macro. Then, when comes the time to check whether the function has
// been overridden, we take the address of the function and we check whether it falls inside
// the special section we created. This can be done by finding pointers to the start and
// the end of the section, and then checking whether `f` falls within those bounds.
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

#  if !__has_feature(ptrauth_calls)

#    define OVERRIDABLE_FUNCTION [[gnu::weak]]

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

#  else // __has_feature(ptrauth_calls)

#    include <cstdint>
#    include <ptrauth.h>

#    if defined(_LIBCPP_OBJECT_FORMAT_MACHO)
#      define OVERRIDABLE_FUNCTION [[gnu::weak, gnu::section("__TEXT,__lcxx_override,regular,pure_instructions")]]
// Declare two dummy bytes and give them these special `__asm` values. These values are
// defined by the linker, which means that referring to `&__lcxx_override_start` will
// effectively refer to the address where the section starts (and same for the end).
extern char __start___lcxx_override __asm("section$start$__TEXT$__lcxx_override");
extern char __stop___lcxx_override __asm("section$end$__TEXT$__lcxx_override");
#    elif defined(_LIBCPP_OBJECT_FORMAT_ELF)
// This is very similar to what we do for Mach-O above. The ELF linker will implicitly define
// variables with those names corresponding to the start and the end of the section.
//
// See https://stackoverflow.com/questions/16552710/how-do-you-get-the-start-and-end-addresses-of-a-custom-elf-section
#      define OVERRIDABLE_FUNCTION [[gnu::weak, gnu::section("__lcxx_override")]]
extern char __start___lcxx_override;
extern char __stop___lcxx_override;
#    endif

_LIBCPP_BEGIN_NAMESPACE_STD
template <typename T, T* _Func>
_LIBCPP_HIDE_FROM_ABI inline bool __is_function_overridden() noexcept {
  uintptr_t __start = reinterpret_cast<uintptr_t>(&__start___lcxx_override);
  uintptr_t __end   = reinterpret_cast<uintptr_t>(&__stop___lcxx_override);
  uintptr_t __ptr   = reinterpret_cast<uintptr_t>(_Func);

  // We must pass a void* to ptrauth_strip since it only accepts a pointer type. Also, in particular,
  // we must NOT pass a function pointer, otherwise we will strip the function pointer, and then attempt
  // to authenticate and re-sign it when casting it to a uintptr_t again, which will fail because we just
  // stripped the function pointer. See rdar://122927845.
  __ptr = reinterpret_cast<uintptr_t>(ptrauth_strip(reinterpret_cast<void*>(__ptr), ptrauth_key_function_pointer));

  // Finally, the function was overridden if it falls outside of the section's bounds.
  return __ptr < __start || __ptr > __end;
}
_LIBCPP_END_NAMESPACE_STD

#  endif // __has_feature(ptrauth_calls)

#else // defined(_LIBCPP_OBJECT_FORMAT_MACHO) || (defined(_LIBCPP_OBJECT_FORMAT_ELF) && !defined(__NVPTX__))

#  define _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION 0
#  define OVERRIDABLE_FUNCTION [[gnu::weak]]

#endif // defined(_LIBCPP_OBJECT_FORMAT_MACHO) || (defined(_LIBCPP_OBJECT_FORMAT_ELF) && !defined(__NVPTX__))

#endif // _LIBCPP_SRC_INCLUDE_OVERRIDABLE_FUNCTION_H
