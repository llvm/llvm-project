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

/// If an IntegralAP is constructed from Memory, it DOES NOT OWN THAT MEMORY.
/// It will NOT copy the memory (unless, of course, copy() is called) and it
/// won't alllocate anything. The allocation should happen via InterpState or
/// Program.
template <bool Signed> class IntegralAP final {
public:
  union {
    uint64_t *Memory = nullptr;
    uint64_t Val;
  };
  uint32_t BitWidth = 0;
  friend IntegralAP<!Signed>;

  template <typename T, bool InputSigned>
  static T truncateCast(const APInt &V) {
    constexpr unsigned BitSize = sizeof(T) * 8;
    if (BitSize >= V.getBitWidth()) {
      APInt Extended;
      if constexpr (InputSigned)
        Extended = V.sext(BitSize);
      else
        Extended = V.zext(BitSize);
      return std::is_signed_v<T> ? Extended.getSExtValue()
                                 : Extended.getZExtValue();
    }

    return std::is_signed_v<T> ? V.trunc(BitSize).getSExtValue()
                               : V.trunc(BitSize).getZExtValue();
  }

  APInt getValue() const {
    if (singleWord())
      return APInt(BitWidth, Val, Signed);
    unsigned NumWords = llvm::APInt::getNumWords(BitWidth);
    return llvm::APInt(BitWidth, NumWords, Memory);
  }

public:
  using AsUnsigned = IntegralAP<false>;

  void take(uint64_t *NewMemory) {
    assert(!singleWord());
    std::memcpy(NewMemory, Memory, numWords() * sizeof(uint64_t));
    Memory = NewMemory;
  }

  void copy(const APInt &V) {
    assert(BitWidth == V.getBitWidth());
    assert(numWords() == V.getNumWords());

    if (V.isSingleWord()) {
      if constexpr (Signed)
        Val = V.getSExtValue();
      else
        Val = V.getZExtValue();
      return;
    }
    assert(Memory);
    std::memcpy(Memory, V.getRawData(), V.getNumWords() * sizeof(uint64_t));
  }

  IntegralAP() = default;
  /// Zeroed, single-word IntegralAP of the given bitwidth.
  IntegralAP(unsigned BitWidth) : Val(0), BitWidth(BitWidth) {
    assert(singleWord());
  }
  IntegralAP(uint64_t *Memory, unsigned BitWidth)
      : Memory(Memory), BitWidth(BitWidth) {}
  IntegralAP(const APInt &V) : BitWidth(V.getBitWidth()) {
    if (V.isSingleWord()) {
      Val = Signed ? V.getSExtValue() : V.getZExtValue();
    } else {
      Memory = const_cast<uint64_t *>(V.getRawData());
    }
  }

  IntegralAP operator-() const { return IntegralAP(-getValue()); }
  bool operator>(const IntegralAP &RHS) const {
    if constexpr (Signed)
      return getValue().sgt(RHS.getValue());
    return getValue().ugt(RHS.getValue());
  }
  bool operator>=(unsigned RHS) const {
    if constexpr (Signed)
      return getValue().sge(RHS);
    return getValue().uge(RHS);
  }
  bool operator<(IntegralAP RHS) const {
    if constexpr (Signed)
      return getValue().slt(RHS.getValue());
    return getValue().ult(RHS.getValue());
  }

  template <typename Ty, typename = std::enable_if_t<std::is_integral_v<Ty>>>
  explicit operator Ty() const {
    return truncateCast<Ty, Signed>(getValue());
  }

  template <typename T> static IntegralAP from(T Value, unsigned NumBits = 0) {
    if (NumBits == 0)
      NumBits = sizeof(T) * 8;
    assert(NumBits > 0);
    assert(APInt::getNumWords(NumBits) == 1);
    APInt Copy = APInt(NumBits, static_cast<uint64_t>(Value), Signed);
    return IntegralAP<Signed>(Copy);
  }

  constexpr uint32_t bitWidth() const { return BitWidth; }
  constexpr unsigned numWords() const { return APInt::getNumWords(BitWidth); }
  constexpr bool singleWord() const { return numWords() == 1; }

  APSInt toAPSInt(unsigned Bits = 0) const {
    if (Bits == 0)
      Bits = bitWidth();

    APInt V = getValue();
    if constexpr (Signed)
      return APSInt(getValue().sext(Bits), !Signed);
    else
      return APSInt(getValue().zext(Bits), !Signed);
  }
  APValue toAPValue(const ASTContext &) const { return APValue(toAPSInt()); }

  bool isZero() const { return getValue().isZero(); }
  bool isPositive() const {
    if constexpr (Signed)
      return getValue().isNonNegative();
    return true;
  }
  bool isNegative() const {
    if constexpr (Signed)
      return !getValue().isNonNegative();
    return false;
  }
  bool isMin() const {
    if constexpr (Signed)
      return getValue().isMinSignedValue();
    return getValue().isMinValue();
  }
  bool isMax() const {
    if constexpr (Signed)
      return getValue().isMaxSignedValue();
    return getValue().isMaxValue();
  }
  static constexpr bool isSigned() { return Signed; }
  bool isMinusOne() const { return Signed && getValue().isAllOnes(); }

  unsigned countLeadingZeros() const { return getValue().countl_zero(); }

  void print(llvm::raw_ostream &OS) const { getValue().print(OS, Signed); }
  std::string toDiagnosticString(const ASTContext &Ctx) const {
    std::string NameStr;
    llvm::raw_string_ostream OS(NameStr);
    print(OS);
    return NameStr;
  }

  IntegralAP truncate(unsigned BitWidth) const {
    if constexpr (Signed)
      return IntegralAP(
          getValue().trunc(BitWidth).sextOrTrunc(this->bitWidth()));
    else
      return IntegralAP(
          getValue().trunc(BitWidth).zextOrTrunc(this->bitWidth()));
  }

  IntegralAP<false> toUnsigned() const {
    return IntegralAP<false>(Memory, BitWidth);
  }

  void bitcastToMemory(std::byte *Dest) const {
    llvm::StoreIntToMemory(getValue(), (uint8_t *)Dest, bitWidth() / 8);
  }

  static void bitcastFromMemory(const std::byte *Src, unsigned BitWidth,
                                IntegralAP *Result) {
    APInt V(BitWidth, static_cast<uint64_t>(0), Signed);
    llvm::LoadIntFromMemory(V, (const uint8_t *)Src, BitWidth / 8);
    Result->copy(V);
  }

  ComparisonCategoryResult compare(const IntegralAP &RHS) const {
    assert(Signed == RHS.isSigned());
    assert(bitWidth() == RHS.bitWidth());
    APInt V1 = getValue();
    APInt V2 = RHS.getValue();
    if constexpr (Signed) {
      if (V1.slt(V2))
        return ComparisonCategoryResult::Less;
      if (V1.sgt(V2))
        return ComparisonCategoryResult::Greater;
      return ComparisonCategoryResult::Equal;
    }

    assert(!Signed);
    if (V1.ult(V2))
      return ComparisonCategoryResult::Less;
    if (V1.ugt(V2))
      return ComparisonCategoryResult::Greater;
    return ComparisonCategoryResult::Equal;
  }

  static bool increment(IntegralAP A, IntegralAP *R) {
    APSInt One(APInt(A.bitWidth(), 1ull, Signed), !Signed);
    return add(A, IntegralAP<Signed>(One), A.bitWidth() + 1, R);
  }

  static bool decrement(IntegralAP A, IntegralAP *R) {
    APSInt One(APInt(A.bitWidth(), 1ull, Signed), !Signed);
    return sub(A, IntegralAP<Signed>(One), A.bitWidth() + 1, R);
  }

  static bool add(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    return CheckAddSubMulUB<std::plus>(A, B, OpBits, R);
  }

  static bool sub(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    return CheckAddSubMulUB<std::minus>(A, B, OpBits, R);
  }

  static bool mul(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    return CheckAddSubMulUB<std::multiplies>(A, B, OpBits, R);
  }

  static bool rem(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    if constexpr (Signed)
      R->copy(A.getValue().srem(B.getValue()));
    else
      R->copy(A.getValue().urem(B.getValue()));
    return false;
  }

  static bool div(IntegralAP A, IntegralAP B, unsigned OpBits, IntegralAP *R) {
    if constexpr (Signed)
      R->copy(A.getValue().sdiv(B.getValue()));
    else
      R->copy(A.getValue().udiv(B.getValue()));
    return false;
  }

  static bool bitAnd(IntegralAP A, IntegralAP B, unsigned OpBits,
                     IntegralAP *R) {
    R->copy(A.getValue() & B.getValue());
    return false;
  }

  static bool bitOr(IntegralAP A, IntegralAP B, unsigned OpBits,
                    IntegralAP *R) {
    R->copy(A.getValue() | B.getValue());
    return false;
  }

  static bool bitXor(IntegralAP A, IntegralAP B, unsigned OpBits,
                     IntegralAP *R) {
    R->copy(A.getValue() ^ B.getValue());
    return false;
  }

  static bool neg(const IntegralAP &A, IntegralAP *R) {
    APInt AI = A.getValue();
    AI.negate();
    R->copy(AI);
    return false;
  }

  static bool comp(IntegralAP A, IntegralAP *R) {
    R->copy(~A.getValue());
    return false;
  }

  static void shiftLeft(const IntegralAP A, const IntegralAP B, unsigned OpBits,
                        IntegralAP *R) {
    *R = IntegralAP(A.getValue().shl(B.getValue().getZExtValue()));
  }

  static void shiftRight(const IntegralAP A, const IntegralAP B,
                         unsigned OpBits, IntegralAP *R) {
    unsigned ShiftAmount = B.getValue().getZExtValue();
    if constexpr (Signed)
      R->copy(A.getValue().ashr(ShiftAmount));
    else
      R->copy(A.getValue().lshr(ShiftAmount));
  }

  // === Serialization support ===
  size_t bytesToSerialize() const {
    assert(BitWidth != 0);
    return sizeof(uint32_t) + (numWords() * sizeof(uint64_t));
  }

  void serialize(std::byte *Buff) const {
    std::memcpy(Buff, &BitWidth, sizeof(uint32_t));
    if (singleWord())
      std::memcpy(Buff + sizeof(uint32_t), &Val, sizeof(uint64_t));
    else {
      std::memcpy(Buff + sizeof(uint32_t), Memory,
                  numWords() * sizeof(uint64_t));
    }
  }

  static uint32_t deserializeSize(const std::byte *Buff) {
    return *reinterpret_cast<const uint32_t *>(Buff);
  }

  static void deserialize(const std::byte *Buff, IntegralAP<Signed> *Result) {
    uint32_t BitWidth = Result->BitWidth;
    assert(BitWidth != 0);
    unsigned NumWords = llvm::APInt::getNumWords(BitWidth);

    if (NumWords == 1)
      std::memcpy(&Result->Val, Buff + sizeof(uint32_t), sizeof(uint64_t));
    else {
      assert(Result->Memory);
      std::memcpy(Result->Memory, Buff + sizeof(uint32_t),
                  NumWords * sizeof(uint64_t));
    }
  }

private:
  template <template <typename T> class Op>
  static bool CheckAddSubMulUB(const IntegralAP &A, const IntegralAP &B,
                               unsigned BitWidth, IntegralAP *R) {
    if constexpr (!Signed) {
      R->copy(Op<APInt>{}(A.getValue(), B.getValue()));
      return false;
    }

    const APSInt &LHS = A.toAPSInt();
    const APSInt &RHS = B.toAPSInt();
    APSInt Value = Op<APSInt>{}(LHS.extend(BitWidth), RHS.extend(BitWidth));
    APSInt Result = Value.trunc(LHS.getBitWidth());
    R->copy(Result);

    return Result.extend(BitWidth) != Value;
  }
};

template <bool Signed>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     IntegralAP<Signed> I) {
  I.print(OS);
  return OS;
}

template <bool Signed>
IntegralAP<Signed> getSwappedBytes(IntegralAP<Signed> F) {
  return F;
}

} // namespace interp
} // namespace clang

#endif
