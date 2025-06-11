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

llvm::StructType *HLSLBufferLayoutBuilder::layOutStruct(
    const RecordType *RT, const llvm::SmallVector<int32_t> *PackOffsets) {

  // check if we already have the layout type for this struct
  // TODO: Do we need to check for matching PackOffsets?
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

  SmallVector<llvm::Type *> Layout;
  SmallVector<const FieldDecl *> DelayLayoutFields;
  CharUnits CurrentOffset = CharUnits::Zero();
  auto LayOutField = [&](QualType FieldType) {
    llvm::Type *LayoutType = layOutType(FieldType);

    const llvm::DataLayout &DL = CGM.getDataLayout();
    CharUnits Size =
        CharUnits::fromQuantity(DL.getTypeSizeInBits(LayoutType) / 8);
    CharUnits Align = CharUnits::fromQuantity(DL.getABITypeAlign(LayoutType));

    if (LayoutType->isAggregateType() ||
        (CurrentOffset % CBufferRowSize) + Size > CBufferRowSize)
      Align = Align.alignTo(CBufferRowSize);

    CharUnits NextOffset = CurrentOffset.alignTo(Align);
    if (NextOffset > CurrentOffset) {
      llvm::Type *Padding = CGM.getTargetCodeGenInfo().getHLSLPadding(
          CGM, NextOffset - CurrentOffset);
      Layout.emplace_back(Padding);
      CurrentOffset = NextOffset;
    }
    Layout.emplace_back(LayoutType);
    CurrentOffset += Size;
  };

  unsigned PackOffsetIndex = 0;
  while (!RecordDecls.empty()) {
    const CXXRecordDecl *RD = RecordDecls.pop_back_val();

    for (const auto *FD : RD->fields()) {
      assert((!PackOffsets || PackOffsetIndex < PackOffsets->size()) &&
             "number of elements in layout struct does not match number of "
             "packoffset annotations");

      // No PackOffset info at all, or have a valid packoffset/register(c#)
      // annotations value -> layout the field.
      const int PO = PackOffsets ? (*PackOffsets)[PackOffsetIndex++] : -1;
      if (PO != -1) {
        LayOutField(FD->getType());
        continue;
      }
      // Have PackOffset info, but there is no packoffset/register(cX)
      // annotation on this field. Delay the layout until after all of the
      // other elements with packoffsets/register(cX) are processed.
      DelayLayoutFields.emplace_back(FD);
    }
  }

  // process delayed layouts
  for (const FieldDecl *FD : DelayLayoutFields)
    LayOutField(FD->getType());

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
  auto *PaddedEltTy =
      llvm::StructType::get(Context, {EltTy, PaddingTy}, /*isPacked=*/true);
  return llvm::StructType::get(
      Context, {llvm::ArrayType::get(PaddedEltTy, Count - 1), EltTy},
      /*IsPacked=*/true);
}

llvm::Type *HLSLBufferLayoutBuilder::layOutType(QualType Ty) {
  if (const auto *AT = CGM.getContext().getAsConstantArrayType(Ty))
    return layOutArray(AT);

  if (Ty->isStructureOrClassType())
    return layOutStruct(Ty->getAsCanonical<RecordType>());

  return CGM.getTypes().ConvertTypeForMem(Ty);
}

} // namespace CodeGen
} // namespace clang
