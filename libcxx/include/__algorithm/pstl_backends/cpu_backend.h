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

  // Cancel the execution of other jobs - they aren't needed anymore
  void __cancel_execution();

  template <class _RandomAccessIterator1,
            class _RandomAccessIterator2,
            class _RandomAccessIterator3,
            class _Compare,
            class _LeafMerge>
  void __parallel_merge(
      _RandomAccessIterator1 __first1,
      _RandomAccessIterator1 __last1,
      _RandomAccessIterator2 __first2,
      _RandomAccessIterator2 __last2,
      _RandomAccessIterator3 __outit,
      _Compare __comp,
      _LeafMerge __leaf_merge);

  TODO: Document the parallel backend
*/

#include <__algorithm/pstl_backends/cpu_backends/any_of.h>
#include <__algorithm/pstl_backends/cpu_backends/fill.h>
#include <__algorithm/pstl_backends/cpu_backends/find_if.h>
#include <__algorithm/pstl_backends/cpu_backends/for_each.h>
#include <__algorithm/pstl_backends/cpu_backends/merge.h>
#include <__algorithm/pstl_backends/cpu_backends/transform.h>

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_H
