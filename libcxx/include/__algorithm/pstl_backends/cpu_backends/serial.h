// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_SERIAL_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_SERIAL_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __par_backend {
inline namespace __serial_cpu_backend {

template <class _RandomAccessIterator, class _Fp>
_LIBCPP_HIDE_FROM_ABI void __parallel_for(_RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f) {
  __f(__first, __last);
}

// TODO: Complete this list

} // namespace __serial_cpu_backend
} // namespace __par_backend

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_SERIAL_H
