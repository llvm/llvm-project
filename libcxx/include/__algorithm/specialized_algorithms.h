//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SPECIALIZED_ALGORITHMS_H
#define _LIBCPP___ALGORITHM_SPECIALIZED_ALGORITHMS_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// FIXME: This should really be an enum
namespace _Algorithm {
  struct __for_each {};
} // namespace _Algorithm

template <class, class>
struct __iterator_pair {};

template <class _Alg, class _Range>
struct __specialized_algorithm {
  static const bool __has_algorithm = false;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_SPECIALIZED_ALGORITHMS_H
