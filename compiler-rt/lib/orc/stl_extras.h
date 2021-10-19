//===-------- stl_extras.h - Useful STL related functions-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_STL_EXTRAS_H
#define ORC_RT_STL_EXTRAS_H

#include <utility>
#include <tuple>

namespace __orc_rt {

namespace detail {

template <typename F, typename Tuple, std::size_t... I>
decltype(auto) apply_tuple_impl(F &&f, Tuple &&t, std::index_sequence<I...>) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

} // end namespace detail

/// Given an input tuple (a1, a2, ..., an), pass the arguments of the
/// tuple variadically to f as if by calling f(a1, a2, ..., an) and
/// return the result.
///
/// FIXME: Switch to std::apply once we can use c++17.
template <typename F, typename Tuple>
decltype(auto) apply_tuple(F &&f, Tuple &&t) {
  using Indices = std::make_index_sequence<
      std::tuple_size<typename std::decay<Tuple>::type>::value>;

  return detail::apply_tuple_impl(std::forward<F>(f), std::forward<Tuple>(t),
                                  Indices{});
}

} // namespace __orc_rt

#endif // ORC_RT_STL_EXTRAS
