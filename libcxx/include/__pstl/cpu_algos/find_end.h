//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_FIND_END_H
#define _LIBCPP___PSTL_CPU_ALGOS_FIND_END_H

#include <__algorithm/find_end.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/find_if.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/convert_to_integral.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_find_end {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator1>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardIterator2 __last2,
             _BinaryPredicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      typedef typename std::iterator_traits<_ForwardIterator1>::difference_type _DifferenceType;
      _DifferenceType __size2 = __last2 - __first2; // The length of the needle to search for.
      if (__size2 == 0) {
        return __last1; // If the needle length is zero, the last iterator is returned.
      }
      _DifferenceType __size1 = __last1 - __first1;
      if (__size1 < __size2) {
        return __last1; // The range is too small to contain the requested number of consecutive elements.
      }
      // Calculate the length of the tail where a potential match cannot start by definition.
      _DifferenceType __crop = __size2 - 1;
      // We're only interested in the range where a potential match can start: [first, last - crop)
      _ForwardIterator1 __last1_cropped = __last1 - __crop;
      // Run a parallel chunked find_if, covering the range where a potential match can start.
      auto __res = __pstl::__parallel_find<_Backend>(
          __first1,
          __last1_cropped,
          [__first2, __last2, __crop, &__pred](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last) {
            // Uncrop the range to allow std::find_end to find a full match, which can go beyond __brick_last.
            _ForwardIterator1 __brick_last_uncropped = __brick_last + __crop;
            // Run a serial std::find_end inside each of the chunks in parallel.
            _ForwardIterator1 __ret = std::find_end(__brick_first, __brick_last_uncropped, __first2, __last2, __pred);
            // The returned iterator is either a match inside [__brick_first, __brick_last) or a miss encoded as
            // __brick_last_uncropped. Return the miss as __brick_last to conform to expectations of __parallel_find().
            return __ret == __brick_last_uncropped ? __brick_last : __ret;
          },
          greater<>{}, // `greater` here means the highest index among the matches
          false        // `false` here means we want the last match, not the first
      );
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      if (*__res == __last1_cropped) {
        return __last1; // No match was found in the range.
      }
      return *__res; // Return the successful match.
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::find_end(
          std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), std::move(__pred));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_FIND_END_H
