//===- HLSLBufferLayoutBuilder.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HLSLBufferLayoutBuilder.h"
#include "CGHLSLRuntime.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Type.h"
#include <climits>

//===----------------------------------------------------------------------===//
// Implementation of constant buffer layout common between DirectX and
// SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

using namespace clang;
using namespace clang::CodeGen;

static const CharUnits CBufferRowSize =
    CharUnits::fromQuantity(llvm::hlsl::CBufferRowSizeInBytes);

namespace clang {
namespace CodeGen {

llvm::StructType *
HLSLBufferLayoutBuilder::layOutStruct(const RecordType *RT,
                                      const CGHLSLOffsetInfo &OffsetInfo) {

  // check if we already have the layout type for this struct
  // TODO: Do we need to check for matching OffsetInfo?
  if (llvm::StructType *Ty = CGM.getHLSLRuntime().getHLSLBufferLayoutType(RT))
    return Ty;

  // iterate over all fields of the record, including fields on base classes
  llvm::SmallVector<CXXRecordDecl *> RecordDecls;
  RecordDecls.push_back(RT->castAsCXXRecordDecl());
  while (RecordDecls.back()->getNumBases()) {
    CXXRecordDecl *D = RecordDecls.back();
    assert(D->getNumBases() == 1 &&
           "HLSL doesn't support multiple inheritance");
    RecordDecls.push_back(D->bases_begin()->getType()->castAsCXXRecordDecl());
  }

  SmallVector<std::pair<const FieldDecl *, uint32_t>> FieldsWithOffset;
  unsigned OffsetIdx = 0;
  for (const CXXRecordDecl *RD : llvm::reverse(RecordDecls))
    for (const auto *FD : RD->fields())
      FieldsWithOffset.emplace_back(FD, OffsetInfo[OffsetIdx++]);

  if (!OffsetInfo.empty())
    llvm::stable_sort(FieldsWithOffset, [](const auto &LHS, const auto &RHS) {
      return CGHLSLOffsetInfo::compareOffsets(LHS.second, RHS.second);
    });

  SmallVector<llvm::Type *> Layout;
  CharUnits CurrentOffset = CharUnits::Zero();
  for (auto &[FD, Offset] : FieldsWithOffset) {
    llvm::Type *LayoutType = layOutType(FD->getType());

    const llvm::DataLayout &DL = CGM.getDataLayout();
    CharUnits Size =
        CharUnits::fromQuantity(DL.getTypeSizeInBits(LayoutType) / 8);
    CharUnits Align = CharUnits::fromQuantity(DL.getABITypeAlign(LayoutType));

    if (LayoutType->isAggregateType() ||
        (CurrentOffset % CBufferRowSize) + Size > CBufferRowSize)
      Align = Align.alignTo(CBufferRowSize);

    CharUnits NextOffset = CurrentOffset.alignTo(Align);

    if (Offset != CGHLSLOffsetInfo::Unspecified) {
      CharUnits PackOffset = CharUnits::fromQuantity(Offset);
      assert(PackOffset >= NextOffset &&
             "Offset is invalid - would overlap with previous object");
      NextOffset = PackOffset;
    }

    if (NextOffset > CurrentOffset) {
      llvm::Type *Padding = CGM.getTargetCodeGenInfo().getHLSLPadding(
          CGM, NextOffset - CurrentOffset);
      assert(Padding && "No padding type for target?");
      Layout.emplace_back(Padding);
      CurrentOffset = NextOffset;
    }
    Layout.emplace_back(LayoutType);
    CurrentOffset += Size;
  }

  // Create the layout struct type; anonymous structs have empty name but
  // non-empty qualified name
  const auto *Decl = RT->castAsCXXRecordDecl();
  std::string Name =
      Decl->getName().empty() ? "anon" : Decl->getQualifiedNameAsString();

  llvm::StructType *NewTy = llvm::StructType::create(Layout, Name,
                                                     /*isPacked=*/true);
  CGM.getHLSLRuntime().addHLSLBufferLayoutType(RT, NewTy);
  return NewTy;
}

llvm::Type *HLSLBufferLayoutBuilder::layOutArray(const ConstantArrayType *AT) {
  llvm::Type *EltTy = layOutType(AT->getElementType());
  uint64_t Count = AT->getZExtSize();

  CharUnits EltSize =
      CharUnits::fromQuantity(CGM.getDataLayout().getTypeSizeInBits(EltTy) / 8);
  CharUnits Padding = EltSize.alignTo(CBufferRowSize) - EltSize;

  // If we don't have any padding between elements then we just need the array
  // itself.
  if (Count < 2 || Padding.isZero())
    return llvm::ArrayType::get(EltTy, Count);

  llvm::LLVMContext &Context = CGM.getLLVMContext();
  llvm::Type *PaddingTy =
      CGM.getTargetCodeGenInfo().getHLSLPadding(CGM, Padding);
  assert(PaddingTy && "No padding type for target?");
  auto *PaddedEltTy =
      llvm::StructType::get(Context, {EltTy, PaddingTy}, /*isPacked=*/true);
  return llvm::StructType::get(
      Context, {llvm::ArrayType::get(PaddedEltTy, Count - 1), EltTy},
      /*IsPacked=*/true);
}

llvm::Type *HLSLBufferLayoutBuilder::layOutType(QualType Ty) {
  if (const auto *AT = CGM.getContext().getAsConstantArrayType(Ty))
    return layOutArray(AT);

  if (Ty->isStructureOrClassType()) {
    CGHLSLOffsetInfo EmptyOffsets;
    return layOutStruct(Ty->getAsCanonical<RecordType>(), EmptyOffsets);
  }

  return CGM.getTypes().ConvertTypeForMem(Ty);
}

} // namespace CodeGen
} // namespace clang
