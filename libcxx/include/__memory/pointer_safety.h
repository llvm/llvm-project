// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_POINTER_SAFETY_H
#define _LIBCPP___MEMORY_POINTER_SAFETY_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

//enum class
#if defined(_LIBCPP_ABI_POINTER_SAFETY_ENUM_TYPE)
# ifndef _LIBCPP_CXX03_LANG
enum class pointer_safety : unsigned char {
  relaxed,
  preferred,
  strict
};
# endif
#else
struct _LIBCPP_TYPE_VIS pointer_safety
{
    enum __lx
    {
        relaxed,
        preferred,
        strict
    };

    __lx __v_;

    _LIBCPP_INLINE_VISIBILITY
    pointer_safety() : __v_() {}

    _LIBCPP_INLINE_VISIBILITY
    pointer_safety(__lx __v) : __v_(__v) {}
    _LIBCPP_INLINE_VISIBILITY
    operator int() const {return __v_;}
};
#endif

#if !defined(_LIBCPP_ABI_POINTER_SAFETY_ENUM_TYPE) && \
    defined(_LIBCPP_BUILDING_LIBRARY)
_LIBCPP_FUNC_VIS pointer_safety get_pointer_safety() _NOEXCEPT;
#else
// This function is only offered in C++03 under ABI v1.
# if !defined(_LIBCPP_ABI_POINTER_SAFETY_ENUM_TYPE) || !defined(_LIBCPP_CXX03_LANG)
inline _LIBCPP_INLINE_VISIBILITY
pointer_safety get_pointer_safety() _NOEXCEPT {
  return pointer_safety::relaxed;
}
# endif
#endif


_LIBCPP_FUNC_VIS void declare_reachable(void* __p);
_LIBCPP_FUNC_VIS void declare_no_pointers(char* __p, size_t __n);
_LIBCPP_FUNC_VIS void undeclare_no_pointers(char* __p, size_t __n);
_LIBCPP_FUNC_VIS void* __undeclare_reachable(void* __p);

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY
_Tp*
undeclare_reachable(_Tp* __p)
{
    return static_cast<_Tp*>(__undeclare_reachable(__p));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_POINTER_SAFETY_H
