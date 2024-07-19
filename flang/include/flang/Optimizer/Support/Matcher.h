//===-- Optimizer/Support/Matcher.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_MATCHER_H
#define FORTRAN_OPTIMIZER_SUPPORT_MATCHER_H

#include "flang/Common/idioms.h"
#include <variant>

// Boilerplate CRTP class for a simplified type-casing syntactic sugar. This
// lets one write pattern matchers using a more compact syntax.
namespace fir::details {
// clang-format off
template<class... Ts> struct matches : Ts... { using Ts::operator()...; };
template<class... Ts> matches(Ts...) -> matches<Ts...>;
template<typename N> struct matcher {
  template<typename... Ts> auto match(Ts... ts) {
    return Fortran::common::visit(matches{ts...}, static_cast<N*>(this)->matchee());
  }
  template<typename... Ts> auto match(Ts... ts) const {
    return Fortran::common::visit(matches{ts...}, static_cast<N const*>(this)->matchee());
  }
};
// clang-format on
} // namespace fir::details

#endif // FORTRAN_OPTIMIZER_SUPPORT_MATCHER_H
