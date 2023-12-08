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
//
//
// With GCC 9.3.0 on a Haswell x86 Ubuntu system, doing out-of-tree builds:
// Before:
//  build:
//   6948.53user 212.48system 27:32.92elapsed 433%CPU
//     (0avgtext+0avgdata 6429568maxresident)k
//   36181912inputs+8943720outputs (3613684major+97908699minor)pagefaults 0swaps
//  execution of tests:
//   205.99user 26.05system 1:08.87elapsed 336%CPU
//     (0avgtext+0avgdata 2671452maxresident)k
//   244432inputs+355464outputs (422major+8746468minor)pagefaults 0swaps
// After:
//  build:
//   6651.91user 182.57system 25:15.73elapsed 450%CPU
//     (0avgtext+0avgdata 6209296maxresident)k
//   17413480inputs+6376360outputs (1567210major+93068230minor)pagefaults 0swaps
//  execution of tests:
//   201.42user 25.91system 1:04.68elapsed 351%CPU
//     (0avgtext+0avgdata 2661424maxresident)k
//   238840inputs+295912outputs (428major+8489300minor)pagefaults 0swaps

#ifndef FORTRAN_COMMON_VISIT_H_
#define FORTRAN_COMMON_VISIT_H_

#include <type_traits>
#include <variant>

namespace Fortran::common {
namespace log2visit {

template <std::size_t LOW, std::size_t HIGH, typename RESULT, typename VISITOR,
    typename... VARIANT>
inline RESULT Log2VisitHelper(
    VISITOR &&visitor, std::size_t which, VARIANT &&...u) {
  if constexpr (LOW == HIGH) {
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
inline auto visit(VISITOR &&visitor, VARIANT &&...u)
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
