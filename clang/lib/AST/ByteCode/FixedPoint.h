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
using APSInt = llvm::APSInt;

/// Wrapper around fixed point types.
class FixedPoint final {
private:
  llvm::APFixedPoint V;

public:
  FixedPoint(llvm::APFixedPoint &&V) : V(std::move(V)) {}
  FixedPoint(llvm::APFixedPoint &V) : V(V) {}
  FixedPoint(APInt V, llvm::FixedPointSemantics Sem) : V(V, Sem) {}
  // This needs to be default-constructible so llvm::endian::read works.
  FixedPoint()
      : V(APInt(0, 0ULL, false),
          llvm::FixedPointSemantics(0, 0, false, false, false)) {}

  static FixedPoint zero(llvm::FixedPointSemantics Sem) {
    return FixedPoint(APInt(Sem.getWidth(), 0ULL, Sem.isSigned()), Sem);
  }

  static FixedPoint from(const APSInt &I, llvm::FixedPointSemantics Sem,
                         bool *Overflow) {
    return FixedPoint(llvm::APFixedPoint::getFromIntValue(I, Sem, Overflow));
  }
  static FixedPoint from(const llvm::APFloat &I, llvm::FixedPointSemantics Sem,
                         bool *Overflow) {
    return FixedPoint(llvm::APFixedPoint::getFromFloatValue(I, Sem, Overflow));
  }

  operator bool() const { return V.getBoolValue(); }
  void print(llvm::raw_ostream &OS) const { OS << V; }

  APValue toAPValue(const ASTContext &) const { return APValue(V); }
  APSInt toAPSInt(unsigned BitWidth = 0) const { return V.getValue(); }

  unsigned bitWidth() const { return V.getWidth(); }
  bool isSigned() const { return V.isSigned(); }
  bool isZero() const { return V.getValue().isZero(); }
  bool isNegative() const { return V.getValue().isNegative(); }
  bool isPositive() const { return V.getValue().isNonNegative(); }
  bool isMin() const {
    return V == llvm::APFixedPoint::getMin(V.getSemantics());
  }
  bool isMinusOne() const { return V.isSigned() && V.getValue() == -1; }

  FixedPoint truncate(unsigned BitWidth) const { return *this; }

  FixedPoint toSemantics(const llvm::FixedPointSemantics &Sem,
                         bool *Overflow) const {
    return FixedPoint(V.convert(Sem, Overflow));
  }
  llvm::FixedPointSemantics getSemantics() const { return V.getSemantics(); }

  llvm::APFloat toFloat(const llvm::fltSemantics *Sem) const {
    return V.convertToFloat(*Sem);
  }

  llvm::APSInt toInt(unsigned BitWidth, bool Signed, bool *Overflow) const {
    return V.convertToInt(BitWidth, Signed, Overflow);
  }

  std::string toDiagnosticString(const ASTContext &Ctx) const {
    return V.toString();
  }

  ComparisonCategoryResult compare(const FixedPoint &Other) const {
    int c = V.compare(Other.V);
    if (c == 0)
      return ComparisonCategoryResult::Equal;
    else if (c < 0)
      return ComparisonCategoryResult::Less;
    return ComparisonCategoryResult::Greater;
  }

  static bool neg(const FixedPoint &A, FixedPoint *R) {
    bool Overflow = false;
    *R = FixedPoint(A.V.negate(&Overflow));
    return Overflow;
  }

  static bool add(const FixedPoint A, const FixedPoint B, unsigned Bits,
                  FixedPoint *R) {
    bool Overflow = false;
    *R = FixedPoint(A.V.add(B.V, &Overflow));
    return Overflow;
  }
  static bool sub(const FixedPoint A, const FixedPoint B, unsigned Bits,
                  FixedPoint *R) {
    bool Overflow = false;
    *R = FixedPoint(A.V.sub(B.V, &Overflow));
    return Overflow;
  }
  static bool mul(const FixedPoint A, const FixedPoint B, unsigned Bits,
                  FixedPoint *R) {
    bool Overflow = false;
    *R = FixedPoint(A.V.mul(B.V, &Overflow));
    return Overflow;
  }
  static bool div(const FixedPoint A, const FixedPoint B, unsigned Bits,
                  FixedPoint *R) {
    bool Overflow = false;
    *R = FixedPoint(A.V.div(B.V, &Overflow));
    return Overflow;
  }

  static bool shiftLeft(const FixedPoint A, const FixedPoint B, unsigned OpBits,
                        FixedPoint *R) {
    unsigned Amt = B.V.getValue().getLimitedValue(OpBits);
    bool Overflow;
    *R = FixedPoint(A.V.shl(Amt, &Overflow));
    return Overflow;
  }
  static bool shiftRight(const FixedPoint A, const FixedPoint B,
                         unsigned OpBits, FixedPoint *R) {
    unsigned Amt = B.V.getValue().getLimitedValue(OpBits);
    bool Overflow;
    *R = FixedPoint(A.V.shr(Amt, &Overflow));
    return Overflow;
  }

  static bool rem(const FixedPoint A, const FixedPoint B, unsigned Bits,
                  FixedPoint *R) {
    llvm_unreachable("Rem doesn't exist for fixed point values");
    return true;
  }
  static bool bitAnd(const FixedPoint A, const FixedPoint B, unsigned Bits,
                     FixedPoint *R) {
    return true;
  }
  static bool bitOr(const FixedPoint A, const FixedPoint B, unsigned Bits,
                    FixedPoint *R) {
    return true;
  }
  static bool bitXor(const FixedPoint A, const FixedPoint B, unsigned Bits,
                     FixedPoint *R) {
    return true;
  }

  static bool increment(const FixedPoint &A, FixedPoint *R) { return true; }
  static bool decrement(const FixedPoint &A, FixedPoint *R) { return true; }
};

inline FixedPoint getSwappedBytes(FixedPoint F) { return F; }

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, FixedPoint F) {
  F.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
