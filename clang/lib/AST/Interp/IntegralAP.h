//===--- Integral.h - Wrapper for numeric types for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the VM types and helpers operating on types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INTEGRAL_AP_H
#define LLVM_CLANG_AST_INTERP_INTEGRAL_AP_H

#include "clang/AST/APValue.h"
#include "clang/AST/ComparisonCategories.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

#include "Primitives.h"

namespace clang {
namespace interp {

using APInt = llvm::APInt;
using APSInt = llvm::APSInt;
template <unsigned Bits, bool Signed> class Integral;

template <bool Signed> class IntegralAP final {
private:
  friend IntegralAP<!Signed>;
  APInt V;

  template <typename T> static T truncateCast(const APInt &V) {
    constexpr unsigned BitSize = sizeof(T) * 8;
    if (BitSize >= V.getBitWidth())
      return std::is_signed_v<T> ? V.getSExtValue() : V.getZExtValue();

    return std::is_signed_v<T> ? V.trunc(BitSize).getSExtValue()
                               : V.trunc(BitSize).getZExtValue();
  }

public:
  using AsUnsigned = IntegralAP<false>;

  template <typename T>
  IntegralAP(T Value, unsigned BitWidth)
      : V(APInt(BitWidth, static_cast<uint64_t>(Value), Signed)) {}

  IntegralAP(APInt V) : V(V) {}
  /// Arbitrary value for uninitialized variables.
  IntegralAP() : IntegralAP(-1, 1024) {}

  IntegralAP operator-() const { return IntegralAP(-V); }
  IntegralAP operator-(const IntegralAP &Other) const {
    return IntegralAP(V - Other.V);
  }
  bool operator>(const IntegralAP &RHS) const {
    if constexpr (Signed)
      return V.ugt(RHS.V);
    return V.sgt(RHS.V);
  }
  bool operator>=(IntegralAP RHS) const {
    if constexpr (Signed)
      return V.uge(RHS.V);
    return V.sge(RHS.V);
  }
  bool operator<(IntegralAP RHS) const {
    if constexpr (Signed)
      return V.slt(RHS.V);
    return V.slt(RHS.V);
  }
  bool operator<=(IntegralAP RHS) const {
    if constexpr (Signed)
      return V.ult(RHS.V);
    return V.ult(RHS.V);
  }

  explicit operator bool() const { return !V.isZero(); }
  explicit operator int8_t() const { return truncateCast<int8_t>(V); }
  explicit operator uint8_t() const { return truncateCast<uint8_t>(V); }
  explicit operator int16_t() const { return truncateCast<int16_t>(V); }
  explicit operator uint16_t() const { return truncateCast<uint16_t>(V); }
  explicit operator int32_t() const { return truncateCast<int32_t>(V); }
  explicit operator uint32_t() const { return truncateCast<uint32_t>(V); }
  explicit operator int64_t() const { return truncateCast<int64_t>(V); }
  explicit operator uint64_t() const { return truncateCast<uint64_t>(V); }

  template <typename T> static IntegralAP from(T Value, unsigned NumBits = 0) {
    assert(NumBits > 0);
    APInt Copy = APInt(NumBits, static_cast<uint64_t>(Value), Signed);

    return IntegralAP<Signed>(Copy);
  }

  template <bool InputSigned>
  static IntegralAP from(IntegralAP<InputSigned> V, unsigned NumBits = 0) {
    return IntegralAP<Signed>(V.V);
  }

  template <unsigned Bits, bool InputSigned>
  static IntegralAP from(Integral<Bits, InputSigned> I, unsigned BitWidth) {
    APInt Copy = APInt(BitWidth, static_cast<uint64_t>(I), InputSigned);

    return IntegralAP<Signed>(Copy);
  }

  static IntegralAP zero(int32_t BitWidth) {
    APInt V = APInt(BitWidth, 0LL, Signed);
    return IntegralAP(V);
  }

  constexpr unsigned bitWidth() const { return V.getBitWidth(); }

  APSInt toAPSInt(unsigned Bits = 0) const { return APSInt(V, Signed); }
  APValue toAPValue() const { return APValue(APSInt(V, Signed)); }

  bool isZero() const { return V.isZero(); }
  bool isPositive() const { return V.isNonNegative(); }
  bool isNegative() const { return !V.isNonNegative(); }
  bool isMin() const { return V.isMinValue(); }
  bool isMax() const { return V.isMaxValue(); }
  static bool isSigned() { return Signed; }
  bool isMinusOne() const { return Signed && V == -1; }

  unsigned countLeadingZeros() const { return V.countl_zero(); }

  void print(llvm::raw_ostream &OS) const { OS << V; }
  std::string toDiagnosticString(const ASTContext &Ctx) const {
    std::string NameStr;
    llvm::raw_string_ostream OS(NameStr);
    print(OS);
    return NameStr;
  }

  IntegralAP truncate(unsigned bitWidth) const {
    assert(false);
    return V;
  }

  IntegralAP<false> toUnsigned() const {
    APInt Copy = V;
    return IntegralAP<false>(Copy);
  }

  ComparisonCategoryResult compare(const IntegralAP &RHS) const {
    assert(Signed == RHS.isSigned());
    assert(bitWidth() == RHS.bitWidth());
    if constexpr (Signed) {
      if (V.slt(RHS.V))
        return ComparisonCategoryResult::Less;
      if (V.sgt(RHS.V))
        return ComparisonCategoryResult::Greater;
      return ComparisonCategoryResult::Equal;
    }

    assert(!Signed);
    if (V.ult(RHS.V))
      return ComparisonCategoryResult::Less;
    if (V.ugt(RHS.V))
      return ComparisonCategoryResult::Greater;
    return ComparisonCategoryResult::Equal;
  }

  static bool increment(IntegralAP A, IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    *R = IntegralAP(A.V - 1);
    return false;
  }

  static bool decrement(IntegralAP A, IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    *R = IntegralAP(A.V - 1);
    return false;
  }

  static bool add(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    return CheckAddUB(A, B, OpBits, R);
  }

  static bool sub(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    /// FIXME: Gotta check if the result fits into OpBits bits.
    return CheckSubUB(A, B, R);
  }

  static bool mul(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    return false;
  }

  static bool rem(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    return false;
  }

  static bool div(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    return false;
  }

  static bool bitAnd(IntegralAP A, IntegralAP B, unsigned OpBits,
                     IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    return false;
  }

  static bool bitOr(IntegralAP A, IntegralAP B, unsigned OpBits,
                    IntegralAP *R) {
    assert(false);
    return false;
  }

  static bool bitXor(IntegralAP A, IntegralAP B, unsigned OpBits,
                     IntegralAP *R) {
    // FIXME: Implement.
    assert(false);
    return false;
  }

  static bool neg(const IntegralAP &A, IntegralAP *R) {
    APInt AI = A.V;
    AI.negate();
    *R = IntegralAP(AI);
    return false;
  }

  static bool comp(IntegralAP A, IntegralAP *R) {
    *R = IntegralAP(~A.V);
    return false;
  }

  static void shiftLeft(const IntegralAP A, const IntegralAP B, unsigned OpBits,
                        IntegralAP *R) {
    *R = IntegralAP(A.V.shl(B.V.getZExtValue()));
  }

  static void shiftRight(const IntegralAP A, const IntegralAP B,
                         unsigned OpBits, IntegralAP *R) {
    *R = IntegralAP(A.V.ashr(B.V.getZExtValue()));
  }

private:
  static bool CheckAddUB(const IntegralAP &A, const IntegralAP &B,
                         unsigned BitWidth, IntegralAP *R) {
    if (!A.isSigned()) {
      R->V = A.V + B.V;
      return false;
    }

    const APSInt &LHS = APSInt(A.V, A.isSigned());
    const APSInt &RHS = APSInt(B.V, B.isSigned());

    APSInt Value(LHS.extend(BitWidth) + RHS.extend(BitWidth), false);
    APSInt Result = Value.trunc(LHS.getBitWidth());
    if (Result.extend(BitWidth) != Value)
      return true;

    R->V = Result;
    return false;
  }
  static bool CheckSubUB(const IntegralAP &A, const IntegralAP &B,
                         IntegralAP *R) {
    R->V = A.V - B.V;
    return false; // Success!
  }
};

template <bool Signed>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     IntegralAP<Signed> I) {
  I.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
