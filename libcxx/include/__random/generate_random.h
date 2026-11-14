// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANDOM_GENERATE_RANDOM_H
#define _LIBCPP___RANDOM_GENERATE_RANDOM_H

#include <__algorithm/ranges_generate.h>
#include <__concepts/invocable.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/invoke.h>
#include <__functional/reference_wrapper.h>
#include <__iterator/concepts.h>
#include <__iterator/distance.h>
#include <__iterator/next.h>
#include <__random/uniform_random_bit_generator.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__ranges/data.h>
#include <__ranges/size.h>
#include <__ranges/subrange.h>
#include <__type_traits/common_type.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <span>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __generate_random {

// [alg.rand.generate]/1.1: g.generate_random(std::forward<R>(r))
template <class _Gen, class _Range>
concept __engine_has_range_member =
    requires(_Gen& __g, _Range&& __r) { __g.generate_random(std::forward<_Range>(__r)); };

// [alg.rand.generate]/1.2: g.generate_random(s), where s is a span<invoke_result_t<G&>>
template <class _Gen, class _Tp>
concept __engine_has_span_member = requires(_Gen& __g, span<_Tp> __s) { __g.generate_random(__s); };

// [alg.rand.generate]/5.1: d.generate_random(std::forward<R>(r), g)
template <class _Dist, class _Range, class _Gen>
concept __dist_has_range_member =
    requires(_Dist& __d, _Range&& __r, _Gen& __g) { __d.generate_random(std::forward<_Range>(__r), __g); };

// [alg.rand.generate]/5.2: d.generate_random(s, g), where s is a span<invoke_result_t<D&, G&>>
template <class _Dist, class _Tp, class _Gen>
concept __dist_has_span_member = requires(_Dist& __d, span<_Tp> __s, _Gen& __g) { __d.generate_random(__s, __g); };

struct __fn {
private:
  // Returns ranges::end(__r) as an iterator, after a generate_random member function has filled __r.
  // See [algorithms.requirements]/13 for converting the sentinel to an iterator.
  template <class _Range>
  static constexpr borrowed_iterator_t<_Range> __end_iterator(_Range& __r) {
    if constexpr (!borrowed_range<_Range>)
      return dangling();
    else if constexpr (common_range<_Range>)
      return ranges::end(__r);
    else
      return ranges::next(ranges::begin(__r), ranges::end(__r));
  }

  // Fills the sized range __r with ranges::size(__r) values. __bulk_gen(__s) fills a span<_Tp> __s with values.
  // If __r is a contiguous range of _Tp, __bulk_gen is called once on all of __r. Otherwise, the values are generated
  // into a buffer on the stack and written to __r one chunk at a time.
  template <class _Tp, class _Range, class _BulkGen>
  static constexpr borrowed_iterator_t<_Range> __fill_in_chunks(_Range& __r, _BulkGen __bulk_gen) {
    if constexpr (contiguous_range<_Range> && is_same_v<range_reference_t<_Range>, _Tp&>) {
      auto __n = ranges::distance(__r);
      if (__n > 0)
        __bulk_gen(span<_Tp>(ranges::data(__r), static_cast<size_t>(__n)));
      return ranges::next(ranges::begin(__r), __n);
    } else {
      // The size is computed in a type that can hold both the size of __r and the size of the buffer. It is computed
      // before calling ranges::begin, which can move the iterator out of __r.
      using _Size                    = common_type_t<range_size_t<_Range>, size_t>;
      _Size __n                      = ranges::size(__r);
      auto __it                      = ranges::begin(__r);
      constexpr size_t __buffer_size = 1024 / sizeof(_Tp);
      _Tp __buffer[__buffer_size];
      while (__n != 0) {
        size_t __chunk = __n < _Size(__buffer_size) ? static_cast<size_t>(__n) : __buffer_size;
        __bulk_gen(span<_Tp>(__buffer, __chunk));
        for (size_t __i = 0; __i != __chunk; ++__i, (void)++__it)
          *__it = std::move(__buffer[__i]);
        __n -= __chunk;
      }
      return __it;
    }
  }

public:
  template <class _Range, class _Gen>
    requires output_range<_Range, invoke_result_t<_Gen&>> && uniform_random_bit_generator<remove_cvref_t<_Gen>>
  static constexpr borrowed_iterator_t<_Range> operator()(_Range&& __r, _Gen&& __g) {
    using _Tp = invoke_result_t<_Gen&>;
    if constexpr (__engine_has_range_member<_Gen, _Range>) {
      __g.generate_random(std::forward<_Range>(__r));
      return __end_iterator<_Range>(__r);
    } else if constexpr (sized_range<_Range> && __engine_has_span_member<_Gen, _Tp>) {
      return __fill_in_chunks<_Tp, _Range>(__r, [&__g](span<_Tp> __s) { __g.generate_random(__s); });
    } else {
      return ranges::generate(std::forward<_Range>(__r), std::ref(__g));
    }
  }

  template <class _Gen, output_iterator<invoke_result_t<_Gen&>> _Out, sentinel_for<_Out> _Sent>
    requires uniform_random_bit_generator<remove_cvref_t<_Gen>>
  static constexpr _Out operator()(_Out __first, _Sent __last, _Gen&& __g) {
    return operator()(subrange<_Out, _Sent>(std::move(__first), std::move(__last)), __g);
  }

  template <class _Range, class _Gen, class _Dist>
    requires output_range<_Range, invoke_result_t<_Dist&, _Gen&>> && invocable<_Dist&, _Gen&> &&
             uniform_random_bit_generator<remove_cvref_t<_Gen>> && is_arithmetic_v<invoke_result_t<_Dist&, _Gen&>>
  static constexpr borrowed_iterator_t<_Range> operator()(_Range&& __r, _Gen&& __g, _Dist&& __d) {
    using _Tp = invoke_result_t<_Dist&, _Gen&>;
    if constexpr (__dist_has_range_member<_Dist, _Range, _Gen>) {
      __d.generate_random(std::forward<_Range>(__r), __g);
      return __end_iterator<_Range>(__r);
    } else if constexpr (sized_range<_Range> && __dist_has_span_member<_Dist, _Tp, _Gen>) {
      return __fill_in_chunks<_Tp, _Range>(__r, [&__d, &__g](span<_Tp> __s) { __d.generate_random(__s, __g); });
    } else {
      return ranges::generate(std::forward<_Range>(__r), [&__d, &__g] { return std::invoke(__d, __g); });
    }
  }

  template <class _Gen, class _Dist, output_iterator<invoke_result_t<_Dist&, _Gen&>> _Out, sentinel_for<_Out> _Sent>
    requires invocable<_Dist&, _Gen&> && uniform_random_bit_generator<remove_cvref_t<_Gen>> &&
             is_arithmetic_v<invoke_result_t<_Dist&, _Gen&>>
  static constexpr _Out operator()(_Out __first, _Sent __last, _Gen&& __g, _Dist&& __d) {
    return operator()(subrange<_Out, _Sent>(std::move(__first), std::move(__last)), __g, __d);
  }
};

} // namespace __generate_random

inline namespace __cpo {
inline constexpr auto generate_random = __generate_random::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANDOM_GENERATE_RANDOM_H
