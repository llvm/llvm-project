//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_H

#include <__config>

/*

  // _Functor takes a subrange for [__first, __last) that should be executed in serial
  template <class _RandomAccessIterator, class _Functor>
  void __parallel_for(_RandomAccessIterator __first, _RandomAccessIterator __last, _Functor __func);

  TODO: Document the parallel backend
*/

#include <__algorithm/pstl_backends/cpu_backends/for_each.h>

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_H
