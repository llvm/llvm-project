//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_RANGES_UNIQUE_H
#define _LIBCPP___CXX03___ALGORITHM_RANGES_UNIQUE_H

#include <__cxx03/__algorithm/iterator_operations.h>
#include <__cxx03/__algorithm/make_projected.h>
#include <__cxx03/__algorithm/unique.h>
#include <__cxx03/__config>
#include <__cxx03/__functional/identity.h>
#include <__cxx03/__functional/invoke.h>
#include <__cxx03/__functional/ranges_operations.h>
#include <__cxx03/__iterator/concepts.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__iterator/permutable.h>
#include <__cxx03/__iterator/projected.h>
#include <__cxx03/__ranges/access.h>
#include <__cxx03/__ranges/concepts.h>
#include <__cxx03/__ranges/dangling.h>
#include <__cxx03/__ranges/subrange.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __unique {

struct __fn {
  template <permutable _Iter,
            sentinel_for<_Iter> _Sent,
            class _Proj                                                  = identity,
            indirect_equivalence_relation<projected<_Iter, _Proj>> _Comp = ranges::equal_to>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr subrange<_Iter>
  operator()(_Iter __first, _Sent __last, _Comp __comp = {}, _Proj __proj = {}) const {
    auto __ret =
        std::__unique<_RangeAlgPolicy>(std::move(__first), std::move(__last), std::__make_projected(__comp, __proj));
    return {std::move(__ret.first), std::move(__ret.second)};
  }

  template <forward_range _Range,
            class _Proj                                                               = identity,
            indirect_equivalence_relation<projected<iterator_t<_Range>, _Proj>> _Comp = ranges::equal_to>
    requires permutable<iterator_t<_Range>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr borrowed_subrange_t<_Range>
  operator()(_Range&& __range, _Comp __comp = {}, _Proj __proj = {}) const {
    auto __ret = std::__unique<_RangeAlgPolicy>(
        ranges::begin(__range), ranges::end(__range), std::__make_projected(__comp, __proj));
    return {std::move(__ret.first), std::move(__ret.second)};
  }
};

} // namespace __unique

inline namespace __cpo {
inline constexpr auto unique = __unique::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ALGORITHM_RANGES_UNIQUE_H
