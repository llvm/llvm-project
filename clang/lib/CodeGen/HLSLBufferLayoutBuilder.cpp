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

// Creates a layout type for given struct or class with HLSL constant buffer
// layout taking into account PackOffsets, if provided.
// Previously created layout types are cached by CGHLSLRuntime.
//
// The function iterates over all fields of the record type (including base
// classes) and calls layoutField to converts each field to its corresponding
// LLVM type and to calculate its HLSL constant buffer layout. Any embedded
// structs (or arrays of structs) are converted to target layout types as well.
//
// When PackOffsets are specified the elements will be placed based on the
// user-specified offsets. Not all elements must have a packoffset/register(c#)
// annotation though. For those that don't, the PackOffsets array will contain
// -1 value instead. These elements must be placed at the end of the layout
// after all of the elements with specific offset.
llvm::TargetExtType *HLSLBufferLayoutBuilder::createLayoutType(
    const RecordType *RT, const llvm::SmallVector<int32_t> *PackOffsets) {

  // check if we already have the layout type for this struct
  if (llvm::TargetExtType *Ty =
          CGM.getHLSLRuntime().getHLSLBufferLayoutType(RT))
    return Ty;

  SmallVector<unsigned> Layout;
  SmallVector<llvm::Type *> LayoutElements;
  unsigned Index = 0; // packoffset index
  unsigned EndOffset = 0;

  SmallVector<std::pair<const FieldDecl *, unsigned>> DelayLayoutFields;

  // reserve first spot in the layout vector for buffer size
  Layout.push_back(0);

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
        Layout.push_back(FieldOffset);
        LayoutElements.push_back(FieldType);
        continue;
      }
      // Have PackOffset info, but there is no packoffset/register(cX)
      // annotation on this field. Delay the layout until after all of the
      // other elements with packoffsets/register(cX) are processed.
      DelayLayoutFields.emplace_back(FD, LayoutElements.size());
      // reserve space for this field in the layout vector and elements list
      Layout.push_back(UINT_MAX);
      LayoutElements.push_back(nullptr);
    }
  }

  // process delayed layouts
  for (auto I : DelayLayoutFields) {
    const FieldDecl *FD = I.first;
    const unsigned IndexInLayoutElements = I.second;
    // the first item in layout vector is size, so we need to offset the index
    // by 1
    const unsigned IndexInLayout = IndexInLayoutElements + 1;
    assert(Layout[IndexInLayout] == UINT_MAX &&
           LayoutElements[IndexInLayoutElements] == nullptr);

    if (!layoutField(FD, EndOffset, FieldOffset, FieldType))
      return nullptr;
    Layout[IndexInLayout] = FieldOffset;
    LayoutElements[IndexInLayoutElements] = FieldType;
  }

  // set the size of the buffer
  Layout[0] = EndOffset;

  // create the layout struct type; anonymous struct have empty name but
  // non-empty qualified name
  const auto *Decl = RT->castAsCXXRecordDecl();
  std::string Name =
      Decl->getName().empty() ? "anon" : Decl->getQualifiedNameAsString();
  llvm::StructType *StructTy =
      llvm::StructType::create(LayoutElements, Name, true);

  // create target layout type
  llvm::TargetExtType *NewLayoutTy = llvm::TargetExtType::get(
      CGM.getLLVMContext(), LayoutTypeName, {StructTy}, Layout);
  if (NewLayoutTy)
    CGM.getHLSLRuntime().addHLSLBufferLayoutType(RT, NewLayoutTy);
  return NewLayoutTy;
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
                                          llvm::Type *&FieldType,
                                          int Packoffset) {

  // Size of element; for arrays this is a size of a single element in the
  // array. Total array size of calculated as (ArrayCount-1) * ArrayStride +
  // ElemSize.
  unsigned ElemSize = 0;
  unsigned ElemOffset = 0;
  unsigned ArrayCount = 1;
  unsigned ArrayStride = 0;

  unsigned NextRowOffset = llvm::alignTo(EndOffset, CBufferRowSizeInBytes);

  llvm::Type *ElemLayoutTy = nullptr;
  QualType FieldTy = FD->getType();

  if (FieldTy->isConstantArrayType()) {
    // Unwrap array to find the element type and get combined array size.
    QualType Ty = FieldTy;
    while (Ty->isConstantArrayType()) {
      auto *ArrayTy = CGM.getContext().getAsConstantArrayType(Ty);
      ArrayCount *= ArrayTy->getSExtSize();
      Ty = ArrayTy->getElementType();
    }
    // For array of structures, create a new array with a layout type
    // instead of the structure type.
    if (Ty->isStructureOrClassType()) {
      llvm::Type *NewTy = cast<llvm::TargetExtType>(
          createLayoutType(Ty->getAsCanonical<RecordType>()));
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
    ArrayStride = llvm::alignTo(ElemSize, CBufferRowSizeInBytes);
    ElemOffset = (Packoffset != -1) ? Packoffset : NextRowOffset;

  } else if (FieldTy->isStructureOrClassType()) {
    // Create a layout type for the structure
    ElemLayoutTy = createLayoutType(
        cast<RecordType>(FieldTy->getAsCanonical<RecordType>()));
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
  FieldOffset = ElemOffset;
  FieldType = ElemLayoutTy;
  return true;
}

} // namespace CodeGen
} // namespace clang
