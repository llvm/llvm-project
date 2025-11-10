//===--- Floating.h - Types for the constexpr VM ----------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_AST_INTERP_FLOATING_H
#define LLVM_CLANG_AST_INTERP_FLOATING_H

#include "Primitives.h"
#include "clang/AST/APValue.h"
#include "llvm/ADT/APFloat.h"

// XXX This is just a debugging help. Setting this to 1 will heap-allocate ALL
// floating values.
#define ALLOCATE_ALL 0

namespace clang {
namespace interp {

using APFloat = llvm::APFloat;
using APSInt = llvm::APSInt;
using APInt = llvm::APInt;

/// If a Floating is constructed from Memory, it DOES NOT OWN THAT MEMORY.
/// It will NOT copy the memory (unless, of course, copy() is called) and it
/// won't alllocate anything. The allocation should happen via InterpState or
/// Program.
class Floating final {
private:
  union {
    uint64_t Val = 0;
    uint64_t *Memory;
  };
  llvm::APFloatBase::Semantics Semantics;

  APFloat getValue() const {
    unsigned BitWidth = bitWidth();
    if (singleWord())
      return APFloat(getSemantics(), APInt(BitWidth, Val));
    unsigned NumWords = numWords();
    return APFloat(getSemantics(),
                   APInt(BitWidth, llvm::ArrayRef(Memory, NumWords)));
  }

public:
  Floating() = default;
  Floating(llvm::APFloatBase::Semantics Semantics)
      : Val(0), Semantics(Semantics) {}
  Floating(const APFloat &F) {

    Semantics = llvm::APFloatBase::SemanticsToEnum(F.getSemantics());
    this->copy(F);
  }
  Floating(uint64_t *Memory, llvm::APFloatBase::Semantics Semantics)
      : Memory(Memory), Semantics(Semantics) {}

  APFloat getAPFloat() const { return getValue(); }

  bool operator<(Floating RHS) const { return getValue() < RHS.getValue(); }
  bool operator>(Floating RHS) const { return getValue() > RHS.getValue(); }
  bool operator<=(Floating RHS) const { return getValue() <= RHS.getValue(); }
  bool operator>=(Floating RHS) const { return getValue() >= RHS.getValue(); }

  APFloat::opStatus convertToInteger(APSInt &Result) const {
    bool IsExact;
    return getValue().convertToInteger(Result, llvm::APFloat::rmTowardZero,
                                       &IsExact);
  }

  void toSemantics(const llvm::fltSemantics *Sem, llvm::RoundingMode RM,
                   Floating *Result) const {
    APFloat Copy = getValue();
    bool LosesInfo;
    Copy.convert(*Sem, RM, &LosesInfo);
    (void)LosesInfo;
    Result->copy(Copy);
  }

  APSInt toAPSInt(unsigned NumBits = 0) const {
    return APSInt(getValue().bitcastToAPInt());
  }
  APValue toAPValue(const ASTContext &) const { return APValue(getValue()); }
  void print(llvm::raw_ostream &OS) const {
    // Can't use APFloat::print() since it appends a newline.
    SmallVector<char, 16> Buffer;
    getValue().toString(Buffer);
    OS << Buffer;
  }
  std::string toDiagnosticString(const ASTContext &Ctx) const {
    std::string NameStr;
    llvm::raw_string_ostream OS(NameStr);
    print(OS);
    return NameStr;
  }

  unsigned bitWidth() const {
    return llvm::APFloatBase::semanticsSizeInBits(getSemantics());
  }
  unsigned numWords() const { return llvm::APInt::getNumWords(bitWidth()); }
  bool singleWord() const {
#if ALLOCATE_ALL
    return false;
#endif
    return numWords() == 1;
  }
  static bool singleWord(const llvm::fltSemantics &Sem) {
#if ALLOCATE_ALL
    return false;
#endif
    return APInt::getNumWords(llvm::APFloatBase::getSizeInBits(Sem)) == 1;
  }
  const llvm::fltSemantics &getSemantics() const {
    return llvm::APFloatBase::EnumToSemantics(Semantics);
  }

  void copy(const APFloat &F) {
    if (singleWord()) {
      Val = F.bitcastToAPInt().getZExtValue();
    } else {
      assert(Memory);
      std::memcpy(Memory, F.bitcastToAPInt().getRawData(),
                  numWords() * sizeof(uint64_t));
    }
  }

  void take(uint64_t *NewMemory) {
    if (singleWord())
      return;

    if (Memory)
      std::memcpy(NewMemory, Memory, numWords() * sizeof(uint64_t));
    Memory = NewMemory;
  }

  bool isSigned() const { return true; }
  bool isNegative() const { return getValue().isNegative(); }
  bool isZero() const { return getValue().isZero(); }
  bool isNonZero() const { return getValue().isNonZero(); }
  bool isMin() const { return getValue().isSmallest(); }
  bool isMinusOne() const { return getValue().isExactlyValue(-1.0); }
  bool isNan() const { return getValue().isNaN(); }
  bool isSignaling() const { return getValue().isSignaling(); }
  bool isInf() const { return getValue().isInfinity(); }
  bool isFinite() const { return getValue().isFinite(); }
  bool isNormal() const { return getValue().isNormal(); }
  bool isDenormal() const { return getValue().isDenormal(); }
  llvm::FPClassTest classify() const { return getValue().classify(); }
  APFloat::fltCategory getCategory() const { return getValue().getCategory(); }

  ComparisonCategoryResult compare(const Floating &RHS) const {
    llvm::APFloatBase::cmpResult CmpRes = getValue().compare(RHS.getValue());
    switch (CmpRes) {
    case llvm::APFloatBase::cmpLessThan:
      return ComparisonCategoryResult::Less;
    case llvm::APFloatBase::cmpEqual:
      return ComparisonCategoryResult::Equal;
    case llvm::APFloatBase::cmpGreaterThan:
      return ComparisonCategoryResult::Greater;
    case llvm::APFloatBase::cmpUnordered:
      return ComparisonCategoryResult::Unordered;
    }
    llvm_unreachable("Inavlid cmpResult value");
  }

  static APFloat::opStatus fromIntegral(APSInt Val,
                                        const llvm::fltSemantics &Sem,
                                        llvm::RoundingMode RM,
                                        Floating *Result) {
    APFloat F = APFloat(Sem);
    APFloat::opStatus Status = F.convertFromAPInt(Val, Val.isSigned(), RM);
    Result->copy(F);
    return Status;
  }

  static void bitcastFromMemory(const std::byte *Buff,
                                const llvm::fltSemantics &Sem,
                                Floating *Result) {
    size_t Size = APFloat::semanticsSizeInBits(Sem);
    llvm::APInt API(Size, true);
    llvm::LoadIntFromMemory(API, (const uint8_t *)Buff, Size / 8);
    Result->copy(APFloat(Sem, API));
  }

  void bitcastToMemory(std::byte *Buff) const {
    llvm::APInt API = getValue().bitcastToAPInt();
    llvm::StoreIntToMemory(API, (uint8_t *)Buff, bitWidth() / 8);
  }

  // === Serialization support ===
  size_t bytesToSerialize() const {
    return sizeof(Semantics) + (numWords() * sizeof(uint64_t));
  }

  void serialize(std::byte *Buff) const {
    std::memcpy(Buff, &Semantics, sizeof(Semantics));
    if (singleWord()) {
      std::memcpy(Buff + sizeof(Semantics), &Val, sizeof(uint64_t));
    } else {
      std::memcpy(Buff + sizeof(Semantics), Memory,
                  numWords() * sizeof(uint64_t));
    }
  }

  static llvm::APFloatBase::Semantics
  deserializeSemantics(const std::byte *Buff) {
    return *reinterpret_cast<const llvm::APFloatBase::Semantics *>(Buff);
  }

  static void deserialize(const std::byte *Buff, Floating *Result) {
    llvm::APFloatBase::Semantics Semantics;
    std::memcpy(&Semantics, Buff, sizeof(Semantics));

    unsigned BitWidth = llvm::APFloat::semanticsSizeInBits(
        llvm::APFloatBase::EnumToSemantics(Semantics));
    unsigned NumWords = llvm::APInt::getNumWords(BitWidth);

    Result->Semantics = Semantics;
    if (NumWords == 1 && !ALLOCATE_ALL) {
      std::memcpy(&Result->Val, Buff + sizeof(Semantics), sizeof(uint64_t));
    } else {
      assert(Result->Memory);
      std::memcpy(Result->Memory, Buff + sizeof(Semantics),
                  NumWords * sizeof(uint64_t));
    }
  }

  // -------

  static APFloat::opStatus add(const Floating &A, const Floating &B,
                               llvm::RoundingMode RM, Floating *R) {
    APFloat LHS = A.getValue();
    APFloat RHS = B.getValue();

    auto Status = LHS.add(RHS, RM);
    R->copy(LHS);
    return Status;
  }

  static APFloat::opStatus increment(const Floating &A, llvm::RoundingMode RM,
                                     Floating *R) {
    APFloat One(A.getSemantics(), 1);
    APFloat LHS = A.getValue();

    auto Status = LHS.add(One, RM);
    R->copy(LHS);
    return Status;
  }

  static APFloat::opStatus sub(const Floating &A, const Floating &B,
                               llvm::RoundingMode RM, Floating *R) {
    APFloat LHS = A.getValue();
    APFloat RHS = B.getValue();

    auto Status = LHS.subtract(RHS, RM);
    R->copy(LHS);
    return Status;
  }

  static APFloat::opStatus decrement(const Floating &A, llvm::RoundingMode RM,
                                     Floating *R) {
    APFloat One(A.getSemantics(), 1);
    APFloat LHS = A.getValue();

    auto Status = LHS.subtract(One, RM);
    R->copy(LHS);
    return Status;
  }

  static APFloat::opStatus mul(const Floating &A, const Floating &B,
                               llvm::RoundingMode RM, Floating *R) {

    APFloat LHS = A.getValue();
    APFloat RHS = B.getValue();

    auto Status = LHS.multiply(RHS, RM);
    R->copy(LHS);
    return Status;
  }

  static APFloat::opStatus div(const Floating &A, const Floating &B,
                               llvm::RoundingMode RM, Floating *R) {
    APFloat LHS = A.getValue();
    APFloat RHS = B.getValue();

    auto Status = LHS.divide(RHS, RM);
    R->copy(LHS);
    return Status;
  }

  static bool neg(const Floating &A, Floating *R) {
    R->copy(-A.getValue());
    return false;
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Floating F);
Floating getSwappedBytes(Floating F);

} // namespace interp
} // namespace clang

#endif
