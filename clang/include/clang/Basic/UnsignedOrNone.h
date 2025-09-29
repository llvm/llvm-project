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

#include "llvm/ADT/ValueOrSentinel.h"

namespace clang {

namespace detail {
struct AdjustAddOne {
  constexpr static unsigned toRepresentation(unsigned Value) {
    return Value + 1;
  }
  constexpr static unsigned fromRepresentation(unsigned Value) {
    return Value - 1;
  }
};
} // namespace detail

using UnsignedOrNone = llvm::ValueOrSentinel<unsigned, 0, detail::AdjustAddOne>;

} // namespace clang

#endif // LLVM_CLANG_BASIC_UNSIGNED_OR_NONE_H
