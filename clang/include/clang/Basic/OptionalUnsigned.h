//===- OptionalUnsigned.h - simple optional index-----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines clang::OptionalUnsigned.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPTIONAL_UNSIGNED_H
#define LLVM_CLANG_BASIC_OPTIONAL_UNSIGNED_H

#include <cassert>
#include <llvm/ADT/STLForwardCompat.h>
#include <optional>

namespace clang {

template <class T> struct OptionalUnsigned {
  using underlying_type =
      typename std::conditional_t<std::is_enum_v<T>, std::underlying_type<T>,
                                  llvm::type_identity<T>>::type;
  static_assert(std::is_unsigned_v<underlying_type>);

  constexpr OptionalUnsigned(std::nullopt_t) : Rep(0) {}
  OptionalUnsigned(T Val) : Rep(static_cast<underlying_type>(Val) + 1) {
    assert(has_value());
  }
  OptionalUnsigned(int) = delete;

  constexpr static OptionalUnsigned
  fromInternalRepresentation(underlying_type Rep) {
    return {std::nullopt, Rep};
  }
  constexpr underlying_type toInternalRepresentation() const { return Rep; }

  constexpr bool has_value() const { return Rep != 0; }

  explicit constexpr operator bool() const { return has_value(); }
  T operator*() const {
    assert(has_value());
    return static_cast<T>(Rep - 1);
  }

  T value_or(T Def) const { return has_value() ? operator*() : Def; }

  friend constexpr bool operator==(OptionalUnsigned LHS, OptionalUnsigned RHS) {
    return LHS && RHS ? *LHS == *RHS : bool(LHS) == bool(RHS);
  }
  friend constexpr bool operator!=(OptionalUnsigned LHS, OptionalUnsigned RHS) {
    return !(LHS == RHS);
  }

  friend constexpr bool operator<(OptionalUnsigned LHS, OptionalUnsigned RHS) {
    return LHS != RHS && (!LHS || (RHS && *LHS < *RHS));
  }
  friend constexpr bool operator<=(OptionalUnsigned LHS, OptionalUnsigned RHS) {
    return LHS == RHS || LHS < RHS;
  }
  friend constexpr bool operator>=(OptionalUnsigned LHS, OptionalUnsigned RHS) {
    return !(LHS < RHS);
  }
  friend constexpr bool operator>(OptionalUnsigned LHS, OptionalUnsigned RHS) {
    return !(LHS <= RHS);
  }

private:
  constexpr OptionalUnsigned(std::nullopt_t, underlying_type Rep) : Rep(Rep) {};

  underlying_type Rep;
};

using UnsignedOrNone = OptionalUnsigned<unsigned>;

} // namespace clang

#endif // LLVM_CLANG_BASIC_OPTIONAL_UNSIGNED_H
