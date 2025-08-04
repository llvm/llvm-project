//===- UnsignedOrNone.h - simple optional index-----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines clang::UnsignedOrNone.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_UNSIGNED_OR_NONE_H
#define LLVM_CLANG_BASIC_UNSIGNED_OR_NONE_H

#include <cassert>
#include <optional>

namespace clang {

struct UnsignedOrNone {
  constexpr UnsignedOrNone(std::nullopt_t) : Rep(0) {}
  UnsignedOrNone(unsigned Val) : Rep(Val + 1) { assert(operator bool()); }
  UnsignedOrNone(int) = delete;

  constexpr static UnsignedOrNone fromInternalRepresentation(unsigned Rep) {
    return {std::nullopt, Rep};
  }
  constexpr unsigned toInternalRepresentation() const { return Rep; }

  explicit constexpr operator bool() const { return Rep != 0; }
  unsigned operator*() const {
    assert(operator bool());
    return Rep - 1;
  }

  friend constexpr bool operator==(UnsignedOrNone LHS, UnsignedOrNone RHS) {
    return LHS.Rep == RHS.Rep;
  }
  friend constexpr bool operator!=(UnsignedOrNone LHS, UnsignedOrNone RHS) {
    return LHS.Rep != RHS.Rep;
  }

private:
  constexpr UnsignedOrNone(std::nullopt_t, unsigned Rep) : Rep(Rep) {};

  unsigned Rep;
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_UNSIGNED_OR_NONE_H
