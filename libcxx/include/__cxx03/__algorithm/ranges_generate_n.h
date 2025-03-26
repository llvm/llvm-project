//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_RANGES_GENERATE_N_H
#define _LIBCPP___CXX03___ALGORITHM_RANGES_GENERATE_N_H

#include <__cxx03/__concepts/constructible.h>
#include <__cxx03/__concepts/invocable.h>
#include <__cxx03/__config>
#include <__cxx03/__functional/identity.h>
#include <__cxx03/__functional/invoke.h>
#include <__cxx03/__iterator/concepts.h>
#include <__cxx03/__iterator/incrementable_traits.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__ranges/access.h>
#include <__cxx03/__ranges/concepts.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __generate_n {

struct __fn {
  template <input_or_output_iterator _OutIter, copy_constructible _Func>
    requires invocable<_Func&> && indirectly_writable<_OutIter, invoke_result_t<_Func&>>
  _LIBCPP_HIDE_FROM_ABI constexpr _OutIter
  operator()(_OutIter __first, iter_difference_t<_OutIter> __n, _Func __gen) const {
    for (; __n > 0; --__n) {
      *__first = __gen();
      ++__first;
    }

    return __first;
  }
};

} // namespace __generate_n

inline namespace __cpo {
inline constexpr auto generate_n = __generate_n::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ALGORITHM_RANGES_GENERATE_N_H
