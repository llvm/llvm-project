//===-- include/flang/Common/visit.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// common::visit() is a drop-in replacement for std::visit() that reduces both
// compiler build time and compiler execution time modestly, and reduces
// compiler build memory requirements significantly (overall & maximum).
// It does not require redefinition of std::variant<>.
//
// The C++ standard mandates that std::visit be O(1), but most variants are
// small and O(logN) is faster in practice to compile and execute, avoiding
// the need to build a dispatch table.
//
// Define FLANG_USE_STD_VISIT to avoid this code and make common::visit() an
// alias for ::std::visit().

#ifndef FORTRAN_COMMON_VISIT_H_
#define FORTRAN_COMMON_VISIT_H_

#include "flang/Common/api-attrs.h"
#include <type_traits>
#include <variant>

namespace Fortran::common {
namespace log2visit {

template <std::size_t LOW, std::size_t HIGH, typename RESULT, typename VISITOR,
    typename... VARIANT>
inline RT_API_ATTRS RESULT Log2VisitHelper(
    VISITOR &&visitor, std::size_t which, VARIANT &&...u) {
  if constexpr (LOW + 7 >= HIGH) {
    switch (which - LOW) {
#define VISIT_CASE_N(N) \
  case N: \
    if constexpr (LOW + N <= HIGH) { \
      return visitor(std::get<(LOW + N)>(std::forward<VARIANT>(u))...); \
    }
      VISIT_CASE_N(1)
      VISIT_CASE_N(2)
      VISIT_CASE_N(3)
      VISIT_CASE_N(4)
      VISIT_CASE_N(5)
      VISIT_CASE_N(6)
      VISIT_CASE_N(7)
#undef VISIT_CASE_N
    }
    return visitor(std::get<LOW>(std::forward<VARIANT>(u))...);
  } else {
    static constexpr std::size_t mid{(HIGH + LOW) / 2};
    if (which <= mid) {
      return Log2VisitHelper<LOW, mid, RESULT>(
          std::forward<VISITOR>(visitor), which, std::forward<VARIANT>(u)...);
    } else {
      return Log2VisitHelper<(mid + 1), HIGH, RESULT>(
          std::forward<VISITOR>(visitor), which, std::forward<VARIANT>(u)...);
    }
  }
}

template <typename VISITOR, typename... VARIANT>
inline RT_API_ATTRS auto visit(VISITOR &&visitor, VARIANT &&...u)
    -> decltype(visitor(std::get<0>(std::forward<VARIANT>(u))...)) {
  using Result = decltype(visitor(std::get<0>(std::forward<VARIANT>(u))...));
  if constexpr (sizeof...(u) == 1) {
    static constexpr std::size_t high{
        (std::variant_size_v<std::decay_t<decltype(u)>> * ...) - 1};
    return Log2VisitHelper<0, high, Result>(std::forward<VISITOR>(visitor),
        u.index()..., std::forward<VARIANT>(u)...);
  } else {
    // TODO: figure out how to do multiple variant arguments
    return ::std::visit(
        std::forward<VISITOR>(visitor), std::forward<VARIANT>(u)...);
  }
}

} // namespace log2visit

// Some versions of clang have bugs that cause compilation to hang
// on these templates.  MSVC and older GCC versions may work but are
// not well tested.  So enable only for GCC 9 and better.
#if __GNUC__ < 9
#define FLANG_USE_STD_VISIT
#endif

#ifdef FLANG_USE_STD_VISIT
using ::std::visit;
#else
using Fortran::common::log2visit::visit;
#endif

} // namespace Fortran::common
#endif // FORTRAN_COMMON_VISIT_H_
