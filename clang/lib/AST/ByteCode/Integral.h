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

#ifndef LLVM_CLANG_AST_INTERP_INTEGRAL_H
#define LLVM_CLANG_AST_INTERP_INTEGRAL_H

#include "clang/AST/APValue.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ComparisonCategories.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

#include "Descriptor.h"
#include "InterpBlock.h"
#include "Primitives.h"

namespace clang {
namespace interp {

using APInt = llvm::APInt;
using APSInt = llvm::APSInt;

template <bool Signed> class IntegralAP;

// Helper structure to select the representation.
template <unsigned Bits, bool Signed> struct Repr;
template <> struct Repr<8, false> {
  using Type = uint8_t;
};
template <> struct Repr<16, false> {
  using Type = uint16_t;
};
template <> struct Repr<32, false> {
  using Type = uint32_t;
};
template <> struct Repr<64, false> {
  using Type = uint64_t;
};
template <> struct Repr<8, true> {
  using Type = int8_t;
};
template <> struct Repr<16, true> {
  using Type = int16_t;
};
template <> struct Repr<32, true> {
  using Type = int32_t;
};
template <> struct Repr<64, true> {
  using Type = int64_t;
};

/// Wrapper around numeric types.
///
/// These wrappers are required to shared an interface between APSint and
/// builtin primitive numeral types, while optimising for storage and
/// allowing methods operating on primitive type to compile to fast code.
template <unsigned Bits, bool Signed> class Integral final {
  static_assert(Bits >= 16);

public:
  // The primitive representing the integral.
  using ReprT = typename Repr<Bits, Signed>::Type;

private:
  using OffsetT = intptr_t;
  static_assert(std::is_trivially_copyable_v<ReprT>);
  template <unsigned OtherBits, bool OtherSigned> friend class Integral;

  IntegralKind Kind = IntegralKind::Number;
  union {
    ReprT V;
    struct {
      const void *P;
      OffsetT Offset;
    } Ptr;
    struct {
      const AddrLabelExpr *L1;
      const AddrLabelExpr *L2;
    } AddrLabelDiff;
  };

  /// Primitive representing limits.
  static const auto Min = std::numeric_limits<ReprT>::min();
  static const auto Max = std::numeric_limits<ReprT>::max();

  /// Construct an integral from anything that is convertible to storage.
  template <typename T> explicit Integral(T V) : V(V) {}
  template <typename T>
  explicit Integral(IntegralKind Kind, T V) : Kind(Kind), V(V) {}

public:
  using AsUnsigned = Integral<Bits, false>;

  /// Zero-initializes an integral.
  Integral() : V(0) {}

  /// Constructs an integral from another integral.
  template <unsigned SrcBits, bool SrcSign>
  explicit Integral(Integral<SrcBits, SrcSign> V) : Kind(V.Kind), V(V) {}

  /// Pointer integral of the given kind.
  explicit Integral(IntegralKind Kind, const void *P, OffsetT Offset = 0)
      : Kind(Kind) {
    Ptr.P = P;
    Ptr.Offset = Offset;
  }

  /// AddrLabelDiff integral.
  explicit Integral(const AddrLabelExpr *P1, const AddrLabelExpr *P2)
      : Kind(IntegralKind::AddrLabelDiff) {
    AddrLabelDiff.L1 = P1;
    AddrLabelDiff.L2 = P2;
  }

  IntegralKind getKind() const { return Kind; }
  bool isNumber() const { return Kind == IntegralKind::Number; }
  const void *getPtr() const {
    assert(!isNumber());
    assert(Kind != IntegralKind::AddrLabelDiff);
    return Ptr.P;
  }
  ReprT getOffset() const {
    assert(!isNumber());
    assert(Kind != IntegralKind::AddrLabelDiff);
    return Ptr.Offset;
  }
  const AddrLabelExpr *getLabel1() const {
    assert(Kind == IntegralKind::AddrLabelDiff);
    return AddrLabelDiff.L1;
  }
  const AddrLabelExpr *getLabel2() const {
    assert(Kind == IntegralKind::AddrLabelDiff);
    return AddrLabelDiff.L2;
  }

  /// Construct an integral from a value based on signedness.
  explicit Integral(const APSInt &V)
      : V(V.isSigned() ? V.getSExtValue() : V.getZExtValue()) {}

  bool operator<(Integral RHS) const { return V < RHS.V; }
  bool operator>(Integral RHS) const { return V > RHS.V; }
  bool operator<=(Integral RHS) const { return V <= RHS.V; }
  bool operator>=(Integral RHS) const { return V >= RHS.V; }
  bool operator==(Integral RHS) const { return V == RHS.V; }
  bool operator!=(Integral RHS) const { return V != RHS.V; }
  bool operator>=(unsigned RHS) const {
    return static_cast<unsigned>(V) >= RHS;
  }

  bool operator>(unsigned RHS) const {
    return V >= 0 && static_cast<unsigned>(V) > RHS;
  }

  Integral operator-() const { return Integral(-V); }
  Integral operator-(const Integral &Other) const {
    return Integral(V - Other.V);
  }
  Integral operator~() const { return Integral(~V); }

  template <unsigned DstBits, bool DstSign>
  explicit operator Integral<DstBits, DstSign>() const {
    return Integral<DstBits, DstSign>(Kind, V);
  }

  template <typename Ty, typename = std::enable_if_t<std::is_integral_v<Ty>>>
  explicit operator Ty() const {
    return V;
  }

  APSInt toAPSInt() const {
    assert(isNumber());
    return APSInt(APInt(Bits, static_cast<uint64_t>(V), Signed), !Signed);
  }

  APSInt toAPSInt(unsigned BitWidth) const {
    return APSInt(toAPInt(BitWidth), !Signed);
  }

  APInt toAPInt(unsigned BitWidth) const {
    assert(isNumber());
    if constexpr (Signed)
      return APInt(Bits, static_cast<uint64_t>(V), Signed)
          .sextOrTrunc(BitWidth);
    else
      return APInt(Bits, static_cast<uint64_t>(V), Signed)
          .zextOrTrunc(BitWidth);
  }

  APValue toAPValue(const ASTContext &) const {
    switch (Kind) {
    case IntegralKind::Address: {
      return APValue((const ValueDecl *)Ptr.P,
                     CharUnits::fromQuantity(Ptr.Offset),
                     APValue::NoLValuePath{});
    }
    case IntegralKind::LabelAddress: {
      return APValue((const Expr *)Ptr.P, CharUnits::Zero(),
                     APValue::NoLValuePath{});
    }
    case IntegralKind::BlockAddress: {
      const Block *B = reinterpret_cast<const Block *>(Ptr.P);
      const Descriptor *D = B->getDescriptor();
      if (const Expr *E = D->asExpr())
        return APValue(E, CharUnits::Zero(), APValue::NoLValuePath{});

      return APValue(D->asValueDecl(), CharUnits::Zero(),
                     APValue::NoLValuePath{});
    }
    case IntegralKind::FunctionAddress: {
      return APValue((const FunctionDecl *)Ptr.P,
                     CharUnits::fromQuantity(Ptr.Offset),
                     APValue::NoLValuePath{});
    }
    case IntegralKind::AddrLabelDiff: {
      return APValue(AddrLabelDiff.L1, AddrLabelDiff.L2);
    }
    case IntegralKind::Number:
      return APValue(toAPSInt());
    }
    llvm_unreachable("Unhandled IntegralKind");
  }

  Integral<Bits, false> toUnsigned() const {
    return Integral<Bits, false>(*this);
  }

  constexpr static unsigned bitWidth() { return Bits; }
  constexpr static bool isSigned() { return Signed; }

  bool isZero() const { return !V; }
  bool isMin() const { return *this == min(bitWidth()); }
  bool isMinusOne() const { return Signed && V == ReprT(-1); }
  bool isNegative() const { return V < ReprT(0); }
  bool isPositive() const { return !isNegative(); }

  ComparisonCategoryResult compare(const Integral &RHS) const {
    return Compare(V, RHS.V);
  }

  void bitcastToMemory(std::byte *Dest) const {
    assert(isNumber());
    std::memcpy(Dest, &V, sizeof(V));
  }

  static Integral bitcastFromMemory(const std::byte *Src, unsigned BitWidth) {
    assert(BitWidth == sizeof(ReprT) * 8);
    ReprT V;

    std::memcpy(&V, Src, sizeof(ReprT));
    return Integral(V);
  }

  std::string toDiagnosticString(const ASTContext &Ctx) const {
    std::string NameStr;
    llvm::raw_string_ostream OS(NameStr);
    OS << V;
    return NameStr;
  }

  unsigned countLeadingZeros() const {
    assert(isNumber());
    if constexpr (!Signed)
      return llvm::countl_zero<ReprT>(V);
    if (isPositive())
      return llvm::countl_zero<typename AsUnsigned::ReprT>(
          static_cast<typename AsUnsigned::ReprT>(V));
    llvm_unreachable("Don't call countLeadingZeros() on negative values.");
  }

  Integral truncate(unsigned TruncBits) const {
    assert(TruncBits >= 1);
    if (TruncBits >= Bits)
      return *this;
    const ReprT BitMask = (ReprT(1) << ReprT(TruncBits)) - 1;
    const ReprT SignBit = ReprT(1) << (TruncBits - 1);
    const ReprT ExtMask = ~BitMask;
    return Integral((V & BitMask) | (Signed && (V & SignBit) ? ExtMask : 0));
  }

  void print(llvm::raw_ostream &OS) const {
    switch (Kind) {
    case IntegralKind::Number:
      OS << V;
      break;
    case IntegralKind::AddrLabelDiff:
      OS << AddrLabelDiff.L1 << " - " << AddrLabelDiff.L2 << " (AddrLabelDiff)";
      break;
    case IntegralKind::Address:
      OS << Ptr.P << " + " << Ptr.Offset << " (Address)";
      break;
    case IntegralKind::BlockAddress:
      OS << Ptr.P << " + " << Ptr.Offset << " (BlockAddress)";
      break;
    case IntegralKind::LabelAddress:
      OS << Ptr.P << " + " << Ptr.Offset << " (LabelAddress)";
      break;
    case IntegralKind::FunctionAddress:
      OS << Ptr.P << " + " << Ptr.Offset << " (FunctionAddress)";
    }
  }

  static Integral min(unsigned NumBits) { return Integral(Min); }
  static Integral max(unsigned NumBits) { return Integral(Max); }
  static Integral zero(unsigned BitWidth = 0) { return from(0); }

  template <typename ValT>
  static std::enable_if_t<!std::is_same_v<ValT, IntegralKind>, Integral>
  from(ValT V, unsigned NumBits = 0) {
    if constexpr (std::is_integral_v<ValT>)
      return Integral(V);
    else
      return Integral(static_cast<Integral::ReprT>(V));
  }

  template <unsigned SrcBits, bool SrcSign>
  static std::enable_if_t<SrcBits != 0, Integral>
  from(Integral<SrcBits, SrcSign> V) {
    switch (V.Kind) {
    case IntegralKind::Number:
      return Integral(V.V);
    case IntegralKind::AddrLabelDiff:
      return Integral(V.getLabel1(), V.getLabel2());
    case IntegralKind::Address:
    case IntegralKind::BlockAddress:
    case IntegralKind::LabelAddress:
    case IntegralKind::FunctionAddress:
      return Integral(V.getKind(), V.getPtr(), V.getOffset());
    }
    llvm_unreachable("Unhandled IntegralKind");
  }

  template <typename T> static Integral from(IntegralKind Kind, T V) {
    return Integral(Kind, V);
  }

  static bool increment(Integral A, Integral *R) {
    assert(A.isNumber());
    return add(A, Integral(ReprT(1)), A.bitWidth(), R);
  }

  static bool decrement(Integral A, Integral *R) {
    assert(A.isNumber());
    return sub(A, Integral(ReprT(1)), A.bitWidth(), R);
  }

  static bool add(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    return CheckAddUB(A.V, B.V, R->V);
  }

  static bool sub(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    return CheckSubUB(A.V, B.V, R->V);
  }

  static bool mul(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    return CheckMulUB(A.V, B.V, R->V);
  }

  static bool rem(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    *R = Integral(A.V % B.V);
    return false;
  }

  static bool div(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    *R = Integral(A.V / B.V);
    return false;
  }

  static bool bitAnd(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    *R = Integral(A.V & B.V);
    return false;
  }

  static bool bitOr(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    *R = Integral(A.V | B.V);
    return false;
  }

  static bool bitXor(Integral A, Integral B, unsigned OpBits, Integral *R) {
    assert(A.isNumber() && B.isNumber());
    *R = Integral(A.V ^ B.V);
    return false;
  }

  static bool neg(Integral A, Integral *R) {
    if (Signed && A.isMin())
      return true;

    *R = -A;
    return false;
  }

  static bool comp(Integral A, Integral *R) {
    *R = Integral(~A.V);
    return false;
  }

  template <unsigned RHSBits, bool RHSSign>
  static void shiftLeft(const Integral A, const Integral<RHSBits, RHSSign> B,
                        unsigned OpBits, Integral *R) {
    *R = Integral::from(A.V << B.V, OpBits);
  }

  template <unsigned RHSBits, bool RHSSign>
  static void shiftRight(const Integral A, const Integral<RHSBits, RHSSign> B,
                         unsigned OpBits, Integral *R) {
    *R = Integral::from(A.V >> B.V, OpBits);
  }
};

template <unsigned Bits, bool Signed>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Integral<Bits, Signed> I) {
  I.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
