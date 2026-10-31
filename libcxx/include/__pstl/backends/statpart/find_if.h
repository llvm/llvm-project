//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_FIND_IF_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_FIND_IF_H

#include <__algorithm/find_if.h>
#include <__assert>
#include <__config>
#include <__execution/is_execution_policy.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/backends/statpart/basis.h>
#include <__utility/empty.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl {
namespace __statpart {

template <class _Index, class _DifferenceType, class _Compare>
_LIBCPP_HIDE_FROM_ABI _Index
__simd_first(_Index __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp) noexcept {
  // Experiments show good block sizes like this
  const _DifferenceType __block_size                                    = 8;
  alignas(__statpart::__lane_size) _DifferenceType __lane[__block_size] = {0};
  while (__end - __begin >= __block_size) {
    _DifferenceType __found = 0;
    _PSTL_PRAGMA_SIMD_REDUCTION(| : __found) for (_DifferenceType __i = __begin; __i < __begin + __block_size; ++__i) {
      const _DifferenceType __t = __comp(__first, __i);
      __lane[__i - __begin]     = __t;
      __found |= __t;
    }
    if (__found) {
      _DifferenceType __i;
      // This will vectorize
      for (__i = 0; __i < __block_size; ++__i) {
        if (__lane[__i]) {
          break;
        }
      }
      return __first + __begin + __i;
    }
    __begin += __block_size;
  }

  // Keep remainder scalar
  while (__begin != __end) {
    if (__comp(__first, __begin)) {
      return __first + __begin;
    }
    ++__begin;
  }
  return __first + __end;
}

} // namespace __statpart

template <class _ExecutionPolicy>
struct __find_if<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      return __statpart::__find(
          std::move(__first),
          std::move(__last),
          [&__policy, &__pred](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            using _FindIfUnseq =
                __pstl::__find_if<__statpart_backend_tag, __remove_parallel_policy_t<_ExecutionPolicy>>;
            auto __res = _FindIfUnseq()(std::__remove_parallel_policy(__policy), __brick_first, __brick_last, __pred);
            _LIBCPP_ASSERT_INTERNAL(__res, "unseq/seq should never try to allocate!");
            return *std::move(__res);
          },
          less<>{},
          true);
    } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                         __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using __diff_t = __iterator_difference_type<_ForwardIterator>;
      return __statpart::__simd_first(
          __first, __diff_t(0), __last - __first, [&__pred](_ForwardIterator __iter, __diff_t __i) {
            return __pred(__iter[__i]);
          });
    } else {
      return std::find_if(std::move(__first), std::move(__last), std::move(__pred));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_FIND_IF_H
