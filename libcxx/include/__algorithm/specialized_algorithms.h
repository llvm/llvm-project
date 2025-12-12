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

namespace _Algorithm {
struct __fill_n {};
struct __for_each {};
} // namespace _Algorithm

template <class>
struct __single_iterator;

template <class, class>
struct __iterator_pair;

template <class>
struct __single_range;

// This struct allows specializing algorithms for specific arguments. This is useful when we know a more efficient
// algorithm implementation for e.g. library-defined iterators. _Alg is one of tags defined inside the _Algorithm
// namespace above. _Ranges is an essentially arbitrary subset of the arguments to the algorithm that are used for
// dispatching. This set is specific to the algorithm: look at each algorithm to see which arguments they use for
// dispatching to specialized algorithms.
//
// A specialization of `__specialized_algorithm` has to define `__has_algorithm` to true for the specialized algorithm
// to be used. This is intended for cases where iterators can do generic unwrapping and forward to a different
// specialization of `__specialized_algorithm`.
//
// If __has_algorithm is true, there has to be an operator() which will get called with the actual arguments to the
// algorithm.
template <class _Alg, class... _Ranges>
struct __specialized_algorithm {
  static const bool __has_algorithm = false;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_SPECIALIZED_ALGORITHMS_H
