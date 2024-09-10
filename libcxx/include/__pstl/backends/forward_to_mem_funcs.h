//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_FORWARD_TO_MEM_FUNCS_H
#define _LIBCPP___PSTL_BACKENDS_FORWARD_TO_MEM_FUNCS_H

#include <__algorithm/copy_move_common.h>
#include <__algorithm/equal.h>
#include <__config>
#include <__pstl/backend_fwd.h>
#include <__pstl/dispatch.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _ExecutionPolicy>
struct __copy<__forward_to_mem_funcs_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _In,
            class _Out,
            enable_if_t<__can_lower_copy_assignment_to_memmove<_In, _Out>::value, int> = 0>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_Out*>
  operator()(_Policy&&, _In* __first, _In* __last, _Out* __result) const noexcept {
    return std::__constexpr_memmove(__result, __first, __element_count(__last - __first));
  }
};

template <class _ExecutionPolicy>
struct __copy_n<__forward_to_mem_funcs_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _In,
            class _Size,
            class _Out,
            enable_if_t<__can_lower_copy_assignment_to_memmove<_In, _Out>::value, int> = 0>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_Out*>
  operator()(_Policy&&, _In* __first, _Size __n, _Out* __result) const noexcept {
    return std::__constexpr_memmove(__result, __first, __element_count(__n));
  }
};

template <class _ExecutionPolicy>
struct __move<__forward_to_mem_funcs_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _In,
            class _Out,
            enable_if_t<__can_lower_move_assignment_to_memmove<_In, _Out>::value, int> = 0>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_Out*>
  operator()(_Policy&&, _In* __first, _In* __last, _Out* __result) const noexcept {
    return std::__constexpr_memmove(__result, __first, __element_count(__last - __first));
  }
};

template <class _ExecutionPolicy>
struct __equal<__forward_to_mem_funcs_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _Tp,
            class _Up,
            class _Pred,
            enable_if_t<__can_lower_to_memcmp_equal<_Tp, _Up, _Pred, __identity, __identity>, int> = 0>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&&, _Tp* __first1, _Tp* __last1, _Up* __first2, _Up* __last2) const noexcept {
    if (__last2 - __first2 != __last1 - __first2)
      return false;
    return std::__constexpr_memcmp_equal(__first1, __first2, __element_count(__last1 - __first1));
  }
};

template <class _ExecutionPolicy>
struct __equal_3leg<__forward_to_mem_funcs_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _Tp,
            class _Up,
            class _Pred,
            enable_if_t<__can_lower_to_memcmp_equal<_Tp, _Up, _Pred, __identity, __identity>, int> = 0>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&&, _Tp* __first1, _Tp* __last1, _Up* __first2) const noexcept {
    return std::__constexpr_memcmp_equal(__first1, __first2, __element_count(__last1 - __first2));
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___PSTL_BACKENDS_FORWARD_TO_MEM_FUNCS_H
