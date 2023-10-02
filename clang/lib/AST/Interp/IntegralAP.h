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
class Boolean;

template <bool Signed> class IntegralAP final {
public:
  APSInt V;

public:
  using AsUnsigned = IntegralAP<false>;

  template <typename T>
  IntegralAP(T Value)
      : V(APInt(sizeof(T) * 8, static_cast<uint64_t>(Value),
                std::is_signed_v<T>)) {}

  IntegralAP(APInt V) : V(V) {}
  IntegralAP(APSInt V) : V(V) {}
  /// Arbitrary value for uninitialized variables.
  IntegralAP() : V(APSInt::getMaxValue(1024, Signed)) {}

  IntegralAP operator-() const { return IntegralAP(-V); }
  bool operator>(IntegralAP RHS) const { return V > RHS.V; }
  bool operator>=(IntegralAP RHS) const { return V >= RHS.V; }
  bool operator<(IntegralAP RHS) const { return V < RHS.V; }
  bool operator<=(IntegralAP RHS) const { return V <= RHS.V; }

  explicit operator bool() const { return !V.isZero(); }
  explicit operator int8_t() const { return V.getSExtValue(); }
  explicit operator uint8_t() const { return V.getZExtValue(); }
  explicit operator int16_t() const { return V.getSExtValue(); }
  explicit operator uint16_t() const { return V.getZExtValue(); }
  explicit operator int32_t() const { return V.getSExtValue(); }
  explicit operator uint32_t() const { return V.getZExtValue(); }
  explicit operator int64_t() const { return V.getSExtValue(); }
  explicit operator uint64_t() const { return V.getZExtValue(); }

  template <typename T> static IntegralAP from(T Value, unsigned NumBits = 0) {
    assert(NumBits > 0);
    APSInt Copy = APSInt(APInt(NumBits, static_cast<int64_t>(Value), Signed), !Signed);

    return IntegralAP<Signed>(Copy);
  }

  template <bool InputSigned>
  static IntegralAP from(IntegralAP<InputSigned> V, unsigned NumBits = 0) {
    if constexpr (Signed == InputSigned)
      return V;

    APSInt Copy = V.V;
    Copy.setIsSigned(Signed);

    return IntegralAP<Signed>(Copy);
  }

  template <unsigned Bits, bool InputSigned>
  static IntegralAP from(Integral<Bits, InputSigned> I) {
    // FIXME: Take bits parameter.
    APSInt Copy =
        APSInt(APInt(128, static_cast<int64_t>(I), InputSigned), !Signed);
    Copy.setIsSigned(Signed);

    assert(Copy.isSigned() == Signed);
    return IntegralAP<Signed>(Copy);
  }
  static IntegralAP from(const Boolean &B) {
    assert(false);
    return IntegralAP::zero();
  }

  static IntegralAP zero() {
    assert(false);
    return IntegralAP(0);
  }

  // FIXME: This can't be static if the bitwidth depends on V.
  static constexpr unsigned bitWidth() { return 128; }

  APSInt toAPSInt(unsigned Bits = 0) const { return V; }
  APValue toAPValue() const { return APValue(V); }

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
    APSInt Copy = V;
    Copy.setIsSigned(false);
    return IntegralAP<false>(Copy);
  }

  ComparisonCategoryResult compare(const IntegralAP &RHS) const {
    return Compare(V, RHS.V);
  }

  static bool increment(IntegralAP A, IntegralAP *R) {
    assert(false);
    *R = IntegralAP(A.V + 1);
    return false;
  }

  static bool decrement(IntegralAP A, IntegralAP *R) {
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
    assert(false);
    // return CheckMulUB(A.V, B.V, R->V);
    return false;
  }

  static bool rem(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    assert(false);
    *R = IntegralAP(A.V % B.V);
    return false;
  }

  static bool div(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    assert(false);
    *R = IntegralAP(A.V / B.V);
    return false;
  }

  static bool bitAnd(IntegralAP A, IntegralAP B, unsigned OpBits,
                     IntegralAP *R) {
    assert(false);
    *R = IntegralAP(A.V & B.V);
    return false;
  }

  static bool bitOr(IntegralAP A, IntegralAP B, unsigned OpBits,
                    IntegralAP *R) {
    assert(false);
    *R = IntegralAP(A.V | B.V);
    return false;
  }

  static bool bitXor(IntegralAP A, IntegralAP B, unsigned OpBits,
                     IntegralAP *R) {
    assert(false);
    *R = IntegralAP(A.V ^ B.V);
    return false;
  }

  static bool neg(const IntegralAP &A, IntegralAP *R) {
    APSInt AI = A.V;

    AI.setIsSigned(Signed);
    *R = IntegralAP(AI);
    return false;
  }

  static bool comp(IntegralAP A, IntegralAP *R) {
    *R = IntegralAP(~A.V);
    return false;
  }

  static void shiftLeft(const IntegralAP A, const IntegralAP B, unsigned OpBits,
                        IntegralAP *R) {
    *R = IntegralAP(A.V << B.V.getZExtValue());
  }

  static void shiftRight(const IntegralAP A, const IntegralAP B,
                         unsigned OpBits, IntegralAP *R) {
    *R = IntegralAP(A.V >> B.V.getZExtValue());
  }

private:
  static bool CheckAddUB(const IntegralAP &A, const IntegralAP &B,
                         unsigned BitWidth, IntegralAP *R) {
    if (!A.isSigned()) {
      R->V = A.V + B.V;
      return false;
    }

    const APSInt &LHS = A.V;
    const APSInt &RHS = B.V;

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
