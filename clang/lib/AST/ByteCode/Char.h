//===------- Char.h - Wrapper for numeric types for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_CHAR_H
#define LLVM_CLANG_AST_INTERP_CHAR_H

#include "Integral.h"
#include <limits>

namespace clang {
namespace interp {

template <unsigned N, bool Signed> class Integral;

template <bool Signed> struct CharRepr;
template <> struct CharRepr<false> {
  using Type = uint8_t;
};
template <> struct CharRepr<true> {
  using Type = int8_t;
};

template <bool Signed> class Char final {
private:
  template <bool OtherSigned> friend class Char;
  using ReprT = typename CharRepr<Signed>::Type;
  ReprT V = 0;
  static_assert(std::is_trivially_copyable_v<ReprT>);

public:
  using AsUnsigned = Char<false>;

  constexpr Char() = default;
  constexpr Char(ReprT V) : V(V) {}
  // constexpr Char(const Char &C) : V(C.V) {}
  explicit Char(const APSInt &V)
      : V(V.isSigned() ? V.getSExtValue() : V.getZExtValue()) {}

  template <typename T> static Char from(T t) {
    return Char(static_cast<ReprT>(t));
  }
  template <typename T> static Char from(T t, unsigned BitWidth) {
    return Char(static_cast<ReprT>(t));
  }

  static bool isSigned() { return Signed; }
  static unsigned bitWidth() { return 8; }
  static bool isNumber() { return true; }
  static Char zero(unsigned BitWidth = 8) { return Char(0); }

  constexpr bool isMin() const {
    return V == std::numeric_limits<ReprT>::min();
  }
  constexpr bool isNegative() const { return Signed && V < 0; }
  constexpr bool isPositive() const { return !isNegative(); }
  constexpr bool isZero() const { return V == 0; }
  constexpr bool isMinusOne() const { return Signed && V == -1; }

  template <typename Ty, typename = std::enable_if_t<std::is_integral_v<Ty>>>
  explicit operator Ty() const {
    return V;
  }

  bool operator<(Char RHS) const { return V < RHS.V; }
  bool operator>(Char RHS) const { return V > RHS.V; }
  bool operator<=(Char RHS) const { return V <= RHS.V; }
  bool operator>=(Char RHS) const { return V >= RHS.V; }
  bool operator==(Char RHS) const { return V == RHS.V; }
  bool operator!=(Char RHS) const { return V != RHS.V; }
  bool operator>=(unsigned RHS) const {
    return static_cast<unsigned>(V) >= RHS;
  }

  bool operator>(unsigned RHS) const {
    return V >= 0 && static_cast<unsigned>(V) > RHS;
  }

  Char operator-() const { return Char(-V); }
  Char operator-(Char Other) const { return Char(V - Other.V); }

  ComparisonCategoryResult compare(Char RHS) const { return Compare(V, RHS.V); }

  void bitcastToMemory(std::byte *Dest) const {
    std::memcpy(Dest, &V, sizeof(V));
  }

  static Char bitcastFromMemory(const std::byte *Src, unsigned BitWidth) {
    assert(BitWidth == 8);
    ReprT V;

    std::memcpy(&V, Src, sizeof(ReprT));
    return Char(V);
  }

  APSInt toAPSInt() const {
    return APSInt(APInt(8, static_cast<uint64_t>(V), Signed), !Signed);
  }
  APSInt toAPSInt(unsigned BitWidth) const {
    return APSInt(toAPInt(BitWidth), !Signed);
  }
  APInt toAPInt(unsigned BitWidth) const {
    if constexpr (Signed)
      return APInt(8, static_cast<uint64_t>(V), Signed).sextOrTrunc(BitWidth);
    else
      return APInt(8, static_cast<uint64_t>(V), Signed).zextOrTrunc(BitWidth);
  }
  APValue toAPValue(const ASTContext &) const { return APValue(toAPSInt()); }
  std::string toDiagnosticString(const ASTContext &Ctx) const {
    std::string NameStr;
    llvm::raw_string_ostream OS(NameStr);
    OS << V;
    return NameStr;
  }
  Char<false> toUnsigned() const { return Char<false>(V); }

  Char truncate(unsigned TruncBits) const {
    assert(TruncBits >= 1);
    if (TruncBits >= 8)
      return *this;
    const ReprT BitMask = (ReprT(1) << ReprT(TruncBits)) - 1;
    const ReprT SignBit = ReprT(1) << (TruncBits - 1);
    const ReprT ExtMask = ~BitMask;
    return Char((V & BitMask) | (Signed && (V & SignBit) ? ExtMask : 0));
  }

  unsigned countLeadingZeros() const {
    if constexpr (!Signed)
      return llvm::countl_zero<ReprT>(V);
    if (isPositive())
      return llvm::countl_zero<typename AsUnsigned::ReprT>(
          static_cast<typename AsUnsigned::ReprT>(V));
    llvm_unreachable("Don't call countLeadingZeros() on negative values.");
  }

  static bool increment(Char A, Char *R) {
    return add(A, Char(ReprT(1)), A.bitWidth(), R);
  }

  static bool decrement(Char A, Char *R) {
    return sub(A, Char(ReprT(1)), A.bitWidth(), R);
  }

  static bool add(Char A, Char B, unsigned OpBits, Char *R) {
    return CheckAddUB(A.V, B.V, R->V);
  }

  static bool sub(Char A, Char B, unsigned OpBits, Char *R) {
    return CheckSubUB(A.V, B.V, R->V);
  }

  static bool mul(Char A, Char B, unsigned OpBits, Char *R) {
    return CheckMulUB(A.V, B.V, R->V);
  }

  static bool rem(Char A, Char B, unsigned OpBits, Char *R) {
    *R = Char(A.V % B.V);
    return false;
  }

  static bool div(Char A, Char B, unsigned OpBits, Char *R) {
    *R = Char(A.V / B.V);
    return false;
  }

  static bool bitAnd(Char A, Char B, unsigned OpBits, Char *R) {
    *R = Char(A.V & B.V);
    return false;
  }

  static bool bitOr(Char A, Char B, unsigned OpBits, Char *R) {
    *R = Char(A.V | B.V);
    return false;
  }

  static bool bitXor(Char A, Char B, unsigned OpBits, Char *R) {
    *R = Char(A.V ^ B.V);
    return false;
  }

  static bool neg(Char A, Char *R) {
    if (Signed && A.isMin())
      return true;

    *R = Char(-A.V);
    return false;
  }

  static bool comp(Char A, Char *R) {
    *R = Char(~A.V);
    return false;
  }

  template <bool RHSSign>
  static void shiftLeft(const Char A, const Char<RHSSign> B, unsigned OpBits,
                        Char *R) {
    *R = Char(A.V << B.V);
  }

  template <bool RHSSign>
  static void shiftRight(const Char A, const Char<RHSSign> B, unsigned OpBits,
                         Char *R) {
    *R = Char(A.V >> B.V);
  }

  void print(llvm::raw_ostream &OS) const { OS << V; }
};

static_assert(sizeof(Char<true>) == 1);
static_assert(sizeof(Char<false>) == 1);

template <bool Signed>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Char<Signed> I) {
  I.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
