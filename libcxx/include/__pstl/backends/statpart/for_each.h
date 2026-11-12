//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_FOR_EACH_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_FOR_EACH_H

#include <__algorithm/for_each.h>
#include <__assert>
#include <__config>
#include <__execution/is_execution_policy.h>
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

template <class _Iterator, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator __simd_for_each(_Iterator __first, _DifferenceType __n, _Function __f) noexcept {
  _PSTL_PRAGMA_SIMD
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __f(__first[__i]);

  return __first + __n;
}

} // namespace __statpart

template <class _ExecutionPolicy>
struct __for_each<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Function>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Function __func) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      return __statpart::__for_each(
          std::move(__first), std::move(__last), [&](auto __partition_first, auto __partition_last) {
            using _ForEachUnseq =
                __pstl::__for_each<__statpart_backend_tag, __remove_parallel_policy_t<_ExecutionPolicy>>;
            [[maybe_unused]] auto __res =
                _ForEachUnseq()(std::__remove_parallel_policy(__policy), __partition_first, __partition_last, __func);
            _LIBCPP_ASSERT_INTERNAL(__res, "unseq/seq should never try to allocate!");
          });
    } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                         __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      __statpart::__simd_for_each(__first, __last - __first, __func);
      return __empty{};
    } else {
      std::for_each(std::move(__first), std::move(__last), std::move(__func));
      return __empty{};
    }
  }
};

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_FOR_EACH_H
