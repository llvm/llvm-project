//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_PSTL_H
#define _LIBCPP___MEMORY_PSTL_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_HAS_EXPERIMENTAL_PSTL && _LIBCPP_STD_VER >= 17

#  include <__iterator/cpp17_iterator_concepts.h>
#  include <__iterator/iterator_traits.h>
#  include <__pstl/backend.h>
#  include <__pstl/dispatch.h>
#  include <__pstl/handle_exception.h>
#  include <__type_traits/enable_if.h>
#  include <__type_traits/is_execution_policy.h>
#  include <__type_traits/remove_cvref.h>
#  include <__utility/forward.h>
#  include <__utility/move.h>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void destroy(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "destroy requires ForwardIterators");
  using _Implementation = __pstl::__dispatch<__pstl::__destroy, __pstl::__current_configuration, _RawPolicy>;
  __pstl::__handle_exception<_Implementation>(
      std::forward<_ExecutionPolicy>(__policy), std::move(__first), std::move(__last));
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void destroy_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _Size __n) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator, "destroy_n requires ForwardIterators");
  using _Implementation = __pstl::__dispatch<__pstl::__destroy_n, __pstl::__current_configuration, _RawPolicy>;
  __pstl::__handle_exception<_Implementation>(std::forward<_ExecutionPolicy>(__policy), std::move(__first), __n);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_EXPERIMENTAL_PSTL && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_PSTL_H
