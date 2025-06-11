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
#include "clang/AST/Type.h"
#include <climits>

//===----------------------------------------------------------------------===//
// Implementation of constant buffer layout common between DirectX and
// SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

using namespace clang;
using namespace clang::CodeGen;
using llvm::hlsl::CBufferRowSizeInBytes;

static unsigned getSizeForCBufferElement(const llvm::DataLayout &DL,
                                         llvm::Type *Ty) {
  // TODO: This is a hack, and it doesn't work for structs containing vectors.
  // We need to get the DataLayout rules correct instead and simply use
  // `getTypeSizeInBits(Ty) / 8` here.
  switch (Ty->getTypeID()) {
  case llvm::Type::ArrayTyID: {
    llvm::ArrayType *ATy = cast<llvm::ArrayType>(Ty);
    return ATy->getNumElements() *
           getSizeForCBufferElement(DL, ATy->getElementType());
  }
  case llvm::Type::FixedVectorTyID: {
    llvm::FixedVectorType *VTy = cast<llvm::FixedVectorType>(Ty);
    return VTy->getNumElements() *
           getSizeForCBufferElement(DL, VTy->getElementType());
  }
  default:
    return DL.getTypeSizeInBits(Ty) / 8;
  }
}

static llvm::Type *createCBufArrayType(llvm::LLVMContext &Context,
                                       const llvm::DataLayout &DL,
                                       llvm::Type *EltTy, unsigned ArrayCount) {
  unsigned EltSize = getSizeForCBufferElement(DL, EltTy);
  unsigned Padding = llvm::alignTo(EltSize, 16) - EltSize;
  // If we don't have any padding between elements then we just need the array
  // itself.
  if (ArrayCount < 2 || !Padding)
    return llvm::ArrayType::get(EltTy, ArrayCount);

  auto *PaddingTy =
      llvm::ArrayType::get(llvm::Type::getInt8Ty(Context), Padding);
  auto *PaddedEltTy =
      llvm::StructType::get(Context, {EltTy, PaddingTy}, /*isPacked=*/true);
  return llvm::StructType::get(
      Context, {llvm::ArrayType::get(PaddedEltTy, ArrayCount - 1), EltTy},
      /*IsPacked=*/true);
}

namespace clang {
namespace CodeGen {

// Creates a layout type for given struct or class with HLSL constant buffer
// layout taking into account PackOffsets, if provided.
// Previously created layout types are cached by CGHLSLRuntime.
//
// The function iterates over all fields of the record type (including base
// classes) and calls layoutField to converts each field to its corresponding
// LLVM type and to calculate its HLSL constant buffer layout. Any embedded
// structs (or arrays of structs) are converted to layout types as well.
//
// When PackOffsets are specified the elements will be placed based on the
// user-specified offsets. Not all elements must have a packoffset/register(c#)
// annotation though. For those that don't, the PackOffsets array will contain
// -1 value instead. These elements must be placed at the end of the layout
// after all of the elements with specific offset.
llvm::StructType *HLSLBufferLayoutBuilder::createLayoutType(
    const RecordType *RT, const llvm::SmallVector<int32_t> *PackOffsets) {

  // check if we already have the layout type for this struct
  if (llvm::StructType *Ty = CGM.getHLSLRuntime().getHLSLBufferLayoutType(RT))
    return Ty;

  SmallVector<std::pair<unsigned, llvm::Type *>> Layout;
  unsigned Index = 0; // packoffset index
  unsigned EndOffset = 0;

  SmallVector<std::pair<const FieldDecl *, unsigned>> DelayLayoutFields;

  // iterate over all fields of the record, including fields on base classes
  llvm::SmallVector<CXXRecordDecl *> RecordDecls;
  RecordDecls.push_back(RT->castAsCXXRecordDecl());
  while (RecordDecls.back()->getNumBases()) {
    CXXRecordDecl *D = RecordDecls.back();
    assert(D->getNumBases() == 1 &&
           "HLSL doesn't support multiple inheritance");
    RecordDecls.push_back(D->bases_begin()->getType()->castAsCXXRecordDecl());
  }

  unsigned FieldOffset;
  llvm::Type *FieldType;

  while (!RecordDecls.empty()) {
    const CXXRecordDecl *RD = RecordDecls.pop_back_val();

    for (const auto *FD : RD->fields()) {
      assert((!PackOffsets || Index < PackOffsets->size()) &&
             "number of elements in layout struct does not match number of "
             "packoffset annotations");

      // No PackOffset info at all, or have a valid packoffset/register(c#)
      // annotations value -> layout the field.
      const int PO = PackOffsets ? (*PackOffsets)[Index++] : -1;
      if (!PackOffsets || PO != -1) {
        if (!layoutField(FD, EndOffset, FieldOffset, FieldType, PO))
          return nullptr;
        Layout.emplace_back(FieldOffset, FieldType);
        continue;
      }
      // Have PackOffset info, but there is no packoffset/register(cX)
      // annotation on this field. Delay the layout until after all of the
      // other elements with packoffsets/register(cX) are processed.
      DelayLayoutFields.emplace_back(FD, Layout.size());
      // reserve space for this field in the layout vector and elements list
      Layout.emplace_back(UINT_MAX, nullptr);
    }
  }

  // process delayed layouts
  for (auto I : DelayLayoutFields) {
    const FieldDecl *FD = I.first;
    assert(Layout[I.second] == std::pair(UINT_MAX, nullptr));

    if (!layoutField(FD, EndOffset, FieldOffset, FieldType))
      return nullptr;
    Layout[I.second] = {FieldOffset, FieldType};
  }

  // TODO: Just do this as we go above...
  // Work out padding so we can create a packed struct for the entire layout.
  SmallVector<llvm::Type *> PaddedElements;
  unsigned CurOffset = 0;
  const llvm::DataLayout &DL = CGM.getDataLayout();
  llvm::Type *PaddingType = llvm::Type::getInt8Ty(CGM.getLLVMContext());
  for (const auto &[Offset, Element] : Layout) {
    assert(Offset >= CurOffset && "Layout out of order?");
    if (unsigned Padding = Offset - CurOffset)
      PaddedElements.push_back(llvm::ArrayType::get(PaddingType, Padding));
    PaddedElements.push_back(Element);
    CurOffset = Offset + getSizeForCBufferElement(DL, Element);
  }

  // Create the layout struct type; anonymous structs have empty name but
  // non-empty qualified name
  const auto *Decl = RT->castAsCXXRecordDecl();
  std::string Name =
      Decl->getName().empty() ? "anon" : Decl->getQualifiedNameAsString();

  llvm::StructType *NewTy = llvm::StructType::create(PaddedElements, Name,
                                                     /*isPacked=*/true);
  CGM.getHLSLRuntime().addHLSLBufferLayoutType(RT, NewTy);
  return NewTy;
}

// The function converts a single field of HLSL Buffer to its corresponding
// LLVM type and calculates it's layout. Any embedded structs (or
// arrays of structs) are converted to target layout types as well.
// The converted type is set to the FieldType parameter, the element
// offset is set to the FieldOffset parameter. The EndOffset (=size of the
// buffer) is also updated accordingly to the offset just after the placed
// element, unless the incoming EndOffset already larger (may happen in case
// of unsorted packoffset annotations).
// Returns true if the conversion was successful.
// The packoffset parameter contains the field's layout offset provided by the
// user or -1 if there was no packoffset (or register(cX)) annotation.
bool HLSLBufferLayoutBuilder::layoutField(const FieldDecl *FD,
                                          unsigned &EndOffset,
                                          unsigned &FieldOffset,
                                          llvm::Type *&FieldLayoutTy,
                                          int Packoffset) {
  unsigned NextRowOffset = llvm::alignTo(EndOffset, CBufferRowSizeInBytes);
  unsigned FieldSize;

  QualType FieldTy = FD->getType();

  if (FieldTy->isConstantArrayType()) {
    // Unwrap array to find the element type and get combined array size.
    QualType Ty = FieldTy;
    unsigned ArrayCount = 1;
    while (Ty->isConstantArrayType()) {
      auto *ArrayTy = CGM.getContext().getAsConstantArrayType(Ty);
      ArrayCount *= ArrayTy->getSExtSize();
      Ty = ArrayTy->getElementType();
    }

    llvm::Type *NewTy;
    if (Ty->isStructureOrClassType()) {
      NewTy = createLayoutType(Ty->getAsCanonical<RecordType>());
      if (!NewTy)
        return false;
    } else
      NewTy = CGM.getTypes().ConvertTypeForMem(Ty);

    FieldLayoutTy = createCBufArrayType(CGM.getLLVMContext(),
                                        CGM.getDataLayout(), NewTy, ArrayCount);
    FieldOffset = (Packoffset != -1) ? Packoffset : NextRowOffset;
    FieldSize = CGM.getDataLayout().getTypeSizeInBits(FieldLayoutTy) / 8;

  } else if (FieldTy->isStructureOrClassType()) {
    // Create a layout type for the structure
    FieldLayoutTy = createLayoutType(
        cast<RecordType>(FieldTy->getAsCanonical<RecordType>()));
    if (!FieldLayoutTy)
      return false;
    FieldOffset = (Packoffset != -1) ? Packoffset : NextRowOffset;
    FieldSize = CGM.getDataLayout().getTypeSizeInBits(FieldLayoutTy) / 8;

  } else {
    // scalar or vector - find element size and alignment
    unsigned Align = 0;
    FieldLayoutTy = CGM.getTypes().ConvertTypeForMem(FieldTy);
    if (FieldLayoutTy->isVectorTy()) {
      // align vectors by sub element size
      const auto *FVT = cast<llvm::FixedVectorType>(FieldLayoutTy);
      unsigned SubElemSize = FVT->getElementType()->getScalarSizeInBits() / 8;
      FieldSize = FVT->getNumElements() * SubElemSize;
      Align = SubElemSize;
    } else {
      assert(FieldLayoutTy->isIntegerTy() ||
             FieldLayoutTy->isFloatingPointTy());
      FieldSize = FieldLayoutTy->getScalarSizeInBits() / 8;
      Align = FieldSize;
    }

    // calculate or get element offset for the vector or scalar
    if (Packoffset != -1) {
      FieldOffset = Packoffset;
    } else {
      FieldOffset = llvm::alignTo(EndOffset, Align);
      // if the element does not fit, move it to the next row
      if (FieldOffset + FieldSize > NextRowOffset)
        FieldOffset = NextRowOffset;
    }
  }

  // Update end offset of the layout; do not update it if the EndOffset
  // is already bigger than the new value (which may happen with unordered
  // packoffset annotations)
  unsigned NewEndOffset = FieldOffset + FieldSize;
  EndOffset = std::max<unsigned>(EndOffset, NewEndOffset);

  return true;
}

} // namespace CodeGen
} // namespace clang
