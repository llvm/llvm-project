//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_APPLY_CV_H
#define _LIBCPP___TYPE_TRAITS_APPLY_CV_H

#include <__config>
#include <__type_traits/copy_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __apply_cv_impl {
  template <class _Up>
  using __apply _LIBCPP_NODEBUG = __copy_cv_t<_Tp, _Up>;
};

template <class _Tp>
struct __apply_cv_impl<_Tp&> {
  template <class _Up>
  using __apply _LIBCPP_NODEBUG = __copy_cv_t<_Tp, _Up>&;
};

template <class _Tp, class _Up>
using __apply_cv_t _LIBCPP_NODEBUG = typename __apply_cv_impl<_Tp>::template __apply<_Up>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_APPLY_CV_H
