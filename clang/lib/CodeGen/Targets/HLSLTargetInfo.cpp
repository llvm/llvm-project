//===- HLSLTargetInto.cpp--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HLSLTargetInfo.h"
#include "CGHLSLRuntime.h"
#include "TargetInfo.h"
#include "clang/AST/DeclCXX.h"

//===----------------------------------------------------------------------===//
// Target codegen info implementation common between DirectX and SPIR/SPIR-V.
//===----------------------------------------------------------------------===//

namespace {

// Creates a new array type with the same dimentions
// but with the new element type.
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

// Returns the size of a scalar or vector in bytes/
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

// Creates a layout type for given struct with HLSL constant buffer layout
// taking into account Packoffsets, if provided.
// Previously created layout types are cached in CGHLSLRuntime because
// TargetCodeGenInto info is cannot store any data
// (CGM.getTargetCodeGenInfo() returns a const reference to TargetCondegenInfo).
//
// The function iterates over all fields of the StructType (including base
// classes), converts each field to its corresponding LLVM type and calculated
// it's HLSL constant bufffer layout (offset and size). Any embedded struct (or
// arrays of structs) are converted to target layout types as well.
llvm::Type *CommonHLSLTargetCodeGenInfo::createHLSLBufferLayoutType(
    CodeGenModule &CGM, const RecordType *StructType,
    const SmallVector<unsigned> *Packoffsets) const {

  // check if we already have the layout type for this struct
  if (llvm::Type *Ty = CGM.getHLSLRuntime().getHLSLBufferLayoutType(StructType))
    return Ty;

  SmallVector<unsigned> Layout;
  SmallVector<llvm::Type *> LayoutElements;
  unsigned Index = 0; // packoffset index
  unsigned EndOffset = 0;
  const unsigned BufferRowAlign = 16U;

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
      // Size of element; for arrays this is a size of a single element in the
      // array. Total array size of calculated as (ArrayCount-1) * ArrayStride +
      // ElemSize.
      unsigned ElemSize = 0;

      unsigned ElemOffset = 0;
      unsigned ArrayCount = 1;
      unsigned ArrayStride = 0;
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
          llvm::Type *NewTy = cast<llvm::TargetExtType>(
              createHLSLBufferLayoutType(CGM, Ty->getAsStructureType()));
          if (!NewTy)
            return nullptr;
          assert(isa<llvm::TargetExtType>(NewTy) && "expected target type");
          ElemSize = cast<llvm::TargetExtType>(NewTy)->getIntParameter(0);
          ElemLayoutTy = createArrayWithNewElementType(
              CGM, cast<ConstantArrayType>(FieldTy.getTypePtr()), NewTy);
        } else {
          // Array of vectors or scalars
          ElemSize = getScalarOrVectorSizeInBytes(
              CGM.getTypes().ConvertTypeForMem(Ty));
          ElemLayoutTy = CGM.getTypes().ConvertTypeForMem(FieldTy);
        }
        ArrayStride = llvm::alignTo(ElemSize, BufferRowAlign);
        ElemOffset =
            Packoffsets != nullptr ? (*Packoffsets)[Index] : NextRowOffset;

      } else if (FieldTy->isStructureType()) {
        // Create a layout type for the structure
        ElemLayoutTy =
            createHLSLBufferLayoutType(CGM, FieldTy->getAsStructureType());
        if (!ElemLayoutTy)
          return nullptr;
        assert(isa<llvm::TargetExtType>(ElemLayoutTy) &&
               "expected target type");
        ElemSize = cast<llvm::TargetExtType>(ElemLayoutTy)->getIntParameter(0);
        ElemOffset =
            Packoffsets != nullptr ? (*Packoffsets)[Index] : NextRowOffset;
      } else {
        // scalar or vector - find element size and alignment
        unsigned Align = 0;
        ElemLayoutTy = CGM.getTypes().ConvertTypeForMem(FieldTy);
        if (ElemLayoutTy->isVectorTy()) {
          // align vectors by sub element size
          const llvm::FixedVectorType *FVT =
              cast<llvm::FixedVectorType>(ElemLayoutTy);
          unsigned SubElemSize =
              FVT->getElementType()->getScalarSizeInBits() / 8;
          ElemSize = FVT->getNumElements() * SubElemSize;
          Align = SubElemSize;
        } else {
          assert(ElemLayoutTy->isIntegerTy() ||
                 ElemLayoutTy->isFloatingPointTy());
          ElemSize = ElemLayoutTy->getScalarSizeInBits() / 8;
          Align = ElemSize;
        }
        // calculate or get element offset for the vector or scalar
        if (Packoffsets != nullptr) {
          ElemOffset = (*Packoffsets)[Index];
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
  llvm::Type *NewLayoutTy = this->getHLSLLayoutType(CGM, StructTy, Layout);
  if (NewLayoutTy)
    CGM.getHLSLRuntime().addHLSLBufferLayoutType(StructType, NewLayoutTy);
  return NewLayoutTy;
}
