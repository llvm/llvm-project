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

//===----------------------------------------------------------------------===//
// Implementation of constant buffer layout common between DirectX and
// SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

using namespace clang;
using namespace clang::CodeGen;

namespace {

// Creates a new array type with the same dimentions but with the new
// element type.
static llvm::Type *
createArrayWithNewElementType(CodeGenModule &CGM,
                              const ConstantArrayType *ArrayType,
                              llvm::Type *NewElemType) {
  const clang::Type *ArrayElemType = ArrayType->getArrayElementTypeNoTypeQual();
  if (ArrayElemType->isConstantArrayType())
    NewElemType = createArrayWithNewElementType(
        CGM, cast<const ConstantArrayType>(ArrayElemType), NewElemType);
  return llvm::ArrayType::get(NewElemType, ArrayType->getSExtSize());
}

// Returns the size of a scalar or vector in bytes
static unsigned getScalarOrVectorSizeInBytes(llvm::Type *Ty) {
  assert(Ty->isVectorTy() || Ty->isIntegerTy() || Ty->isFloatingPointTy());
  if (Ty->isVectorTy()) {
    llvm::FixedVectorType *FVT = cast<llvm::FixedVectorType>(Ty);
    return FVT->getNumElements() *
           (FVT->getElementType()->getScalarSizeInBits() / 8);
  }
  return Ty->getScalarSizeInBits() / 8;
}

} // namespace

namespace clang {
namespace CodeGen {

// Creates a layout type for given struct with HLSL constant buffer layout
// taking into account Packoffsets, if provided.
// Previously created layout types are cached by CGHLSLRuntime.
//
// The function iterates over all fields of the StructType (including base
// classes) and calls layoutField to converts each field to its corresponding
// LLVM type and to calculate its HLSL constant buffer layout. Any embedded
// structs (or arrays of structs) are converted to target layout types as well.
llvm::TargetExtType *HLSLBufferLayoutBuilder::createLayoutType(
    const RecordType *StructType,
    const llvm::SmallVector<unsigned> *Packoffsets) {

  // check if we already have the layout type for this struct
  if (llvm::TargetExtType *Ty =
          CGM.getHLSLRuntime().getHLSLBufferLayoutType(StructType))
    return Ty;

  SmallVector<unsigned> Layout;
  SmallVector<llvm::Type *> LayoutElements;
  unsigned Index = 0; // packoffset index
  unsigned EndOffset = 0;

  // reserve first spot in the layout vector for buffer size
  Layout.push_back(0);

  // iterate over all fields of the record, including fields on base classes
  llvm::SmallVector<const RecordType *> RecordTypes;
  RecordTypes.push_back(StructType);
  while (RecordTypes.back()->getAsCXXRecordDecl()->getNumBases()) {
    CXXRecordDecl *D = RecordTypes.back()->getAsCXXRecordDecl();
    assert(D->getNumBases() == 1 &&
           "HLSL doesn't support multiple inheritance");
    RecordTypes.push_back(D->bases_begin()->getType()->getAs<RecordType>());
  }
  while (!RecordTypes.empty()) {
    const RecordType *RT = RecordTypes.back();
    RecordTypes.pop_back();

    for (const auto *FD : RT->getDecl()->fields()) {
      assert(!Packoffsets || Index < Packoffsets->size() &&
                                 "number of elements in layout struct does not "
                                 "match number of packoffset annotations");

      if (!layoutField(FD, EndOffset, Layout, LayoutElements,
                       Packoffsets ? (*Packoffsets)[Index] : -1))
        return nullptr;
      Index++;
    }
  }

  // set the size of the buffer
  Layout[0] = EndOffset;

  // create the layout struct type; anonymous struct have empty name but
  // non-empty qualified name
  const CXXRecordDecl *Decl = StructType->getAsCXXRecordDecl();
  std::string Name =
      Decl->getName().empty() ? "anon" : Decl->getQualifiedNameAsString();
  llvm::StructType *StructTy =
      llvm::StructType::create(LayoutElements, Name, true);

  // create target layout type
  llvm::TargetExtType *NewLayoutTy = llvm::TargetExtType::get(
      CGM.getLLVMContext(), LayoutTypeName, {StructTy}, Layout);
  if (NewLayoutTy)
    CGM.getHLSLRuntime().addHLSLBufferLayoutType(StructType, NewLayoutTy);
  return NewLayoutTy;
}

// The function converts a single field of HLSL Buffer to its corresponding
// LLVM type and calculates it's layout. Any embedded structs (or
// arrays of structs) are converted to target layout types as well.
// The converted type is appended to the LayoutElements list, the element
// offset is added to the Layout list and the EndOffset updated to the offset
// just after the lay-ed out element (which is basically the size of the
// buffer).
// Returns true if the conversion was successful.
// The packoffset parameter contains the field's layout offset provided by the
// user or -1 if there was no packoffset (or register(cX)) annotation.
bool HLSLBufferLayoutBuilder::layoutField(
    const FieldDecl *FD, unsigned &EndOffset, SmallVector<unsigned> &Layout,
    SmallVector<llvm::Type *> &LayoutElements, int Packoffset) {

  // Size of element; for arrays this is a size of a single element in the
  // array. Total array size of calculated as (ArrayCount-1) * ArrayStride +
  // ElemSize.
  unsigned ElemSize = 0;
  unsigned ElemOffset = 0;
  unsigned ArrayCount = 1;
  unsigned ArrayStride = 0;

  const unsigned BufferRowAlign = 16U;
  unsigned NextRowOffset = llvm::alignTo(EndOffset, BufferRowAlign);

  llvm::Type *ElemLayoutTy = nullptr;
  QualType FieldTy = FD->getType();

  if (FieldTy->isConstantArrayType()) {
    // Unwrap array to find the element type and get combined array size.
    QualType Ty = FieldTy;
    while (Ty->isConstantArrayType()) {
      const ConstantArrayType *ArrayTy = cast<ConstantArrayType>(Ty);
      ArrayCount *= ArrayTy->getSExtSize();
      Ty = ArrayTy->getElementType();
    }
    // For array of structures, create a new array with a layout type
    // instead of the structure type.
    if (Ty->isStructureType()) {
      llvm::Type *NewTy =
          cast<llvm::TargetExtType>(createLayoutType(Ty->getAsStructureType()));
      if (!NewTy)
        return false;
      assert(isa<llvm::TargetExtType>(NewTy) && "expected target type");
      ElemSize = cast<llvm::TargetExtType>(NewTy)->getIntParameter(0);
      ElemLayoutTy = createArrayWithNewElementType(
          CGM, cast<ConstantArrayType>(FieldTy.getTypePtr()), NewTy);
    } else {
      // Array of vectors or scalars
      ElemSize =
          getScalarOrVectorSizeInBytes(CGM.getTypes().ConvertTypeForMem(Ty));
      ElemLayoutTy = CGM.getTypes().ConvertTypeForMem(FieldTy);
    }
    ArrayStride = llvm::alignTo(ElemSize, BufferRowAlign);
    ElemOffset = (Packoffset != -1) ? Packoffset : NextRowOffset;

  } else if (FieldTy->isStructureType()) {
    // Create a layout type for the structure
    ElemLayoutTy = createLayoutType(FieldTy->getAsStructureType());
    if (!ElemLayoutTy)
      return false;
    assert(isa<llvm::TargetExtType>(ElemLayoutTy) && "expected target type");
    ElemSize = cast<llvm::TargetExtType>(ElemLayoutTy)->getIntParameter(0);
    ElemOffset = (Packoffset != -1) ? Packoffset : NextRowOffset;

  } else {
    // scalar or vector - find element size and alignment
    unsigned Align = 0;
    ElemLayoutTy = CGM.getTypes().ConvertTypeForMem(FieldTy);
    if (ElemLayoutTy->isVectorTy()) {
      // align vectors by sub element size
      const llvm::FixedVectorType *FVT =
          cast<llvm::FixedVectorType>(ElemLayoutTy);
      unsigned SubElemSize = FVT->getElementType()->getScalarSizeInBits() / 8;
      ElemSize = FVT->getNumElements() * SubElemSize;
      Align = SubElemSize;
    } else {
      assert(ElemLayoutTy->isIntegerTy() || ElemLayoutTy->isFloatingPointTy());
      ElemSize = ElemLayoutTy->getScalarSizeInBits() / 8;
      Align = ElemSize;
    }

    // calculate or get element offset for the vector or scalar
    if (Packoffset != -1) {
      ElemOffset = Packoffset;
    } else {
      ElemOffset = llvm::alignTo(EndOffset, Align);
      // if the element does not fit, move it to the next row
      if (ElemOffset + ElemSize > NextRowOffset)
        ElemOffset = NextRowOffset;
    }
  }

  // Update end offset of the layout; do not update it if the EndOffset
  // is already bigger than the new value (which may happen with unordered
  // packoffset annotations)
  unsigned NewEndOffset =
      ElemOffset + (ArrayCount - 1) * ArrayStride + ElemSize;
  EndOffset = std::max<unsigned>(EndOffset, NewEndOffset);

  // add the layout element and offset to the lists
  Layout.push_back(ElemOffset);
  LayoutElements.push_back(ElemLayoutTy);
  return true;
}

} // namespace CodeGen
} // namespace clang
