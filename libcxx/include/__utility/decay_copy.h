// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_DECAY_COPY_H
#define _LIBCPP___UTILITY_DECAY_COPY_H

#include <__config>
#include <__utility/forward.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
inline _LIBCPP_INLINE_VISIBILITY typename decay<_Tp>::type __decay_copy(_Tp&& __t)
#if _LIBCPP_STD_VER > 17
    noexcept(is_nothrow_convertible_v<_Tp, remove_reference_t<_Tp> >)
#endif
{
  return _VSTD::forward<_Tp>(__t);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_DECAY_COPY_H
