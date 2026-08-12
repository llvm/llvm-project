//===--- CodeGenUtils.h - Shared Classic CodeGen/CIR CodeGen Utils--C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"

#include <algorithm>
#include <utility>

namespace clang::CodeGenUtils {
// A type that helps represent a padding interval.
struct BitInterval {
  // [First, Last)
  uint64_t First;
  uint64_t Last;
};

// PaddingCalculator is a utility class that calculates the padding bits in a
// c/c++ type. It traverses the type recursively, collecting occupied
// bit intervals, and then computes the padding intervals.
// If a byte only contains some padding bits, it gets intervals for only those
// bits. This is the case for bit-fields.
struct PaddingCalculator {
  PaddingCalculator(const ASTContext &Ctx, const TargetInfo &TI,
                    unsigned PointerSizeInBits)
      : Ctx(Ctx), TI(TI), PointerSizeInBits(PointerSizeInBits) {}

  void run(QualType Ty) {
    OccuppiedIntervals.clear();
    Stack.clear();

    TySizeInBits = Ctx.getTypeSize(Ty);

    Stack.push_back(Data{0, Ty, true});
    while (!Stack.empty()) {
      Data Current = Stack.back();
      Stack.pop_back();
      Visit(Current);
    }
    MergeOccuppiedIntervals();
  }

  llvm::SmallVector<BitInterval> GetPaddingIntervals() {
    llvm::SmallVector<BitInterval> Results;
    if (OccuppiedIntervals.size() == 1 &&
        OccuppiedIntervals.front().First == 0 &&
        OccuppiedIntervals.front().Last == TySizeInBits) {
      return Results;
    }
    Results.reserve(OccuppiedIntervals.size() + 1);
    uint64_t CurrentPos = 0;
    for (const BitInterval &OccupiedInterval : OccuppiedIntervals) {
      if (OccupiedInterval.First > CurrentPos) {
        Results.push_back(BitInterval{CurrentPos, OccupiedInterval.First});
      }
      CurrentPos = OccupiedInterval.Last;
    }
    if (TySizeInBits > CurrentPos) {
      Results.push_back(BitInterval{CurrentPos, TySizeInBits});
    }
    return Results;
  }

private:
  struct Data {
    uint64_t StartBitOffset;
    QualType Ty;
    bool VisitVirtualBase;
  };

  // Return the number of non padding bits of a scalar type.
  //
  // The property that we specifically care about here is whether the scalar
  // type has padding bits, i.e. are there bits in the type which are not
  // specified by the ABI.
  //
  // We currently don't care about this anywhere else in clang: layout cares
  // about the ABI size, calling convention code cares about specific types,
  // but nothing cares about padding specifically. And it's not something we can
  // easily query from LLVM due to the type system mismatches.
  // DL.getTypeSizeInBits(convertTypeForLoadStore(T)) is probably close, but the
  // DataLayout methods aren't really designed for this usage.
  //
  // Therefore, it is better to explicitly list all the scalar types
  // containing padding bits that we know of, namely, _BitInt(N) and x87 long
  // double.
  uint64_t getScalarOccupiedSizeInBits(QualType Ty) const {
    if (const auto *BIT = Ty->getAs<BitIntType>())
      return BIT->getNumBits();

    if (const auto *BT = Ty->getAs<BuiltinType>()) {
      if (BT->getKind() == BuiltinType::LongDouble &&
          &TI.getLongDoubleFormat() == &llvm::APFloat::x87DoubleExtended())
        return llvm::APFloat::getSizeInBits(TI.getLongDoubleFormat());
    }

    return Ctx.getTypeSize(Ty);
  }

  void Visit(const Data &D) {
    if (auto *AT = dyn_cast<ConstantArrayType>(D.Ty)) {
      VisitArray(AT, D.StartBitOffset);
      return;
    }

    if (auto *Record = D.Ty->getAsRecordDecl()) {
      VisitStruct(Record, D.StartBitOffset, D.VisitVirtualBase);
      return;
    }

    if (D.Ty->isAtomicType()) {
      auto Unwrapped = D;
      Unwrapped.Ty = D.Ty.getAtomicUnqualifiedType();
      Stack.push_back(Unwrapped);
      return;
    }

    if (const auto *Complex = D.Ty->getAs<ComplexType>()) {
      VisitComplex(Complex, D.StartBitOffset);
      return;
    }

    if (const auto *VT = D.Ty->getAs<clang::VectorType>()) {
      VisitVector(VT, D.StartBitOffset);
      return;
    }

    uint64_t SizeBit = getScalarOccupiedSizeInBits(D.Ty);
    OccuppiedIntervals.push_back(
        BitInterval{D.StartBitOffset, D.StartBitOffset + SizeBit});
  }

  void VisitArray(const ConstantArrayType *AT, uint64_t StartBitOffset) {
    for (uint64_t ArrIndex = 0; ArrIndex < AT->getSize().getLimitedValue();
         ++ArrIndex) {

      QualType ElementQualType = AT->getElementType();
      auto ElementSize = Ctx.getTypeSizeInChars(ElementQualType);
      auto ElementAlign = Ctx.getTypeAlignInChars(ElementQualType);
      auto Offset = ElementSize.alignTo(ElementAlign);

      Stack.push_back(Data{StartBitOffset + ArrIndex * Offset.getQuantity() *
                                                Ctx.getCharWidth(),
                           ElementQualType, /*VisitVirtualBase*/ true});
    }
  }

  void VisitStruct(const RecordDecl *R, uint64_t StartBitOffset,
                   bool VisitVirtualBase) {
    const ASTRecordLayout &ASTLayout = Ctx.getASTRecordLayout(R);
    auto *CXXRecord = dyn_cast<CXXRecordDecl>(R);

    if (CXXRecord) {
      if (ASTLayout.hasOwnVFPtr()) {
        OccuppiedIntervals.push_back(
            BitInterval{StartBitOffset, StartBitOffset + PointerSizeInBits});
      }

      if (ASTLayout.hasOwnVBPtr()) {
        auto Offset = ASTLayout.getVBPtrOffset().getQuantity();
        auto StartVBPtr = StartBitOffset + Offset * Ctx.getCharWidth();
        OccuppiedIntervals.push_back(
            BitInterval{StartVBPtr, StartVBPtr + PointerSizeInBits});
      }

      const auto VisitBase = [&ASTLayout, StartBitOffset, this](
                                 const CXXBaseSpecifier &Base, auto GetOffset) {
        auto *BaseRecord = Base.getType()->getAsCXXRecordDecl();
        if (!BaseRecord) {
          return;
        }
        auto BaseOffset =
            std::invoke(GetOffset, ASTLayout, BaseRecord).getQuantity();

        Stack.push_back(Data{StartBitOffset + BaseOffset * Ctx.getCharWidth(),
                             Base.getType(), /*VisitVirtualBase*/ false});
      };

      for (auto Base : CXXRecord->bases()) {
        if (!Base.isVirtual()) {
          VisitBase(Base, &ASTRecordLayout::getBaseClassOffset);
        }
      }

      if (VisitVirtualBase) {
        for (auto VBase : CXXRecord->vbases()) {
          VisitBase(VBase, &ASTRecordLayout::getVBaseClassOffset);
        }
      }
    }

    for (auto *Field : R->fields()) {
      // Treat unnamed bitfields as padding.
      if (Field->isUnnamedBitField())
        continue;

      auto FieldOffset = ASTLayout.getFieldOffset(Field->getFieldIndex());
      if (Field->isBitField()) {
        OccuppiedIntervals.push_back(BitInterval{
            StartBitOffset + FieldOffset,
            StartBitOffset + FieldOffset + Field->getBitWidthValue()});
      } else {
        Stack.push_back(Data{StartBitOffset + FieldOffset, Field->getType(),
                             /*VisitVirtualBase*/ true});
      }
    }
  }

  void VisitComplex(const ComplexType *CT, uint64_t StartBitOffset) {
    QualType ElementQualType = CT->getElementType();
    auto ElementSize = Ctx.getTypeSizeInChars(ElementQualType);
    auto ElementAlign = Ctx.getTypeAlignInChars(ElementQualType);
    auto ImgOffset = ElementSize.alignTo(ElementAlign);

    Stack.push_back(
        Data{StartBitOffset, ElementQualType, /*VisitVirtualBase*/ true});
    Stack.push_back(
        Data{StartBitOffset + ImgOffset.getQuantity() * Ctx.getCharWidth(),
             ElementQualType, /*VisitVirtualBase*/ true});
  }

  void VisitVector(const clang::VectorType *VT, uint64_t StartBitOffset) {
    uint64_t SizeBit = [&]() -> uint64_t {
      if (VT->isPackedVectorBoolType(Ctx))
        return VT->getNumElements();
      return getScalarOccupiedSizeInBits(VT->getElementType()) *
             VT->getNumElements();
    }();
    OccuppiedIntervals.push_back(
        BitInterval{StartBitOffset, StartBitOffset + SizeBit});
  }

  void MergeOccuppiedIntervals() {
    std::sort(OccuppiedIntervals.begin(), OccuppiedIntervals.end(),
              [](const BitInterval &lhs, const BitInterval &rhs) {
                return std::tie(lhs.First, lhs.Last) <
                       std::tie(rhs.First, rhs.Last);
              });

    llvm::SmallVector<BitInterval> Merged;
    Merged.reserve(OccuppiedIntervals.size());

    for (const BitInterval &NextInterval : OccuppiedIntervals) {
      if (Merged.empty()) {
        Merged.push_back(NextInterval);
        continue;
      }
      auto &LastInterval = Merged.back();

      if (NextInterval.First > LastInterval.Last) {
        Merged.push_back(NextInterval);
      } else {
        LastInterval.Last = std::max(LastInterval.Last, NextInterval.Last);
      }
    }

    OccuppiedIntervals = Merged;
  }

  const ASTContext &Ctx;
  const TargetInfo &TI;
  unsigned PointerSizeInBits;
  uint64_t TySizeInBits = 0;
  llvm::SmallVector<Data> Stack;
  llvm::SmallVector<BitInterval> OccuppiedIntervals;
};

// Calculate and gets the 'padding intervals' inside of a type.
llvm::SmallVector<BitInterval>
CalculatePaddingIntervals(const ASTContext &Ctx, const TargetInfo &TI,
                          QualType Ty, unsigned PointerSizeInBits) {
  PaddingCalculator pc{Ctx, TI, PointerSizeInBits};
  pc.run(Ty);
  return pc.GetPaddingIntervals();
}
} // namespace clang::CodeGenUtils
