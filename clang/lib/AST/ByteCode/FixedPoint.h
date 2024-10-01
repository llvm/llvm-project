//===------- FixedPoint.h - Fixedd point types for the VM -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_FIXED_POINT_H
#define LLVM_CLANG_AST_INTERP_FIXED_POINT_H

#include "clang/AST/APValue.h"
#include "clang/AST/ComparisonCategories.h"
#include "llvm/ADT/APFixedPoint.h"

namespace clang {
namespace interp {

using APInt = llvm::APInt;

/// Wrapper around fixed point types.
class FixedPoint final {
private:
  llvm::APFixedPoint V;

public:
  FixedPoint(APInt V)
      : V(V,
          llvm::FixedPointSemantics(V.getBitWidth(), 0, false, false, false)) {}
  // This needs to be default-constructible so llvm::endian::read works.
  FixedPoint()
      : V(APInt(0, 0ULL, false),
          llvm::FixedPointSemantics(0, 0, false, false, false)) {}

  operator bool() const { return V.getBoolValue(); }
  template <typename Ty, typename = std::enable_if_t<std::is_integral_v<Ty>>>
  explicit operator Ty() const {
    // FIXME
    return 0;
  }

  void print(llvm::raw_ostream &OS) const { OS << V; }

  APValue toAPValue(const ASTContext &) const { return APValue(V); }

  ComparisonCategoryResult compare(const FixedPoint &Other) const {
    if (Other.V == V)
      return ComparisonCategoryResult::Equal;
    return ComparisonCategoryResult::Unordered;
  }
};

inline FixedPoint getSwappedBytes(FixedPoint F) { return F; }

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, FixedPoint F) {
  F.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
