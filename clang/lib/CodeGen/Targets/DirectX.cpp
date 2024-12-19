//===- DirectX.cpp---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "llvm/IR/DerivedTypes.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// Target codegen info implementation for DirectX.
//===----------------------------------------------------------------------===//

namespace {

class DirectXTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  DirectXTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<DefaultABIInfo>(CGT)) {}

  llvm::Type *
  getHLSLType(CodeGenModule &CGM, const Type *T,
              const HLSLBufferDecl *BufDecl = nullptr) const override;
};

static bool hasPackoffset(const HLSLBufferDecl *BufferDecl) {
  // If one valid constant buffer declaration has a packoffset annotation
  // then they all need to have them.
  for (auto *D : BufferDecl->decls())
    if (VarDecl *VD = dyn_cast<VarDecl>(D))
      if (VD->getStorageClass() != SC_Static)
        return VD->hasAttr<HLSLPackOffsetAttr>();
  return false;
}

// Fills in the Layout based on the constant buffer packoffset annotations
// found in the HLSLBufferDecl instance.
// The first item in the Layout vector will be set to the size of the buffer,
// which is the highest packoffset value + size of its field.
// After the size there will be a list of offsets (in bytes) for each field.
// This layout encoding will be used in the LLVM target type
// for constant buffers target("dx.CBuffer",..).
static void createConstantBufferLayoutFromPackoffset(
    CodeGenModule &CGM, llvm::Type *BufferTy, SmallVector<unsigned> &Layout,
    const HLSLBufferDecl *BufferDecl) {
  assert(BufferDecl && hasPackoffset(BufferDecl) &&
         "expected non-null buffer declaration with packoffset attributes on "
         "its fields");
  assert(isa<llvm::StructType>(BufferTy) && "expected struct");

  // total buffer size - equals to the highest packoffset value plus the size of
  // its element
  unsigned Size = 0;
  // reserve the first spot in the layout vector for the buffer size
  Layout.push_back(UINT_MAX);

  auto DeclIt = BufferDecl->decls_begin();
  for (llvm::Type *Ty : cast<llvm::StructType>(BufferTy)->elements()) {
    assert(!Ty->isArrayTy() && !Ty->isStructTy() &&
           "arrays and structs in cbuffer are not yet implemened");

    VarDecl *VD = nullptr;
    // ignore any declaration that is not var decls or that is static
    while (DeclIt != BufferDecl->decls_end()) {
      VD = llvm::dyn_cast<VarDecl>(*DeclIt);
      DeclIt++;
      if (VD && VD->getStorageClass() != SC_Static)
        break;
    }
    assert(VD && "number of declaration in constant buffer does not match "
                 "number of element in buffer struct");
    assert(CGM.getTypes().ConvertType(VD->getType()) == Ty &&
           "constant types do not match");

    HLSLPackOffsetAttr *POAttr = VD->getAttr<HLSLPackOffsetAttr>();
    assert(POAttr && "expected packoffset attribute on every declaration");

    unsigned Offset = POAttr->getOffsetInBytes();
    Layout.push_back(Offset);

    unsigned FieldSize = Ty->getScalarSizeInBits() / 8;
    if (Offset + FieldSize > Size)
      Size = Offset + FieldSize;
  }
  Layout[0] = Size;
}

// Calculated constant buffer layout. The first item in the Layout
// vector will be set to the size of the buffer followed by the offsets
// (in bytes) for each field.
// This layout encoding will be used in the LLVM target type
// for constant buffers target("dx.CBuffer",..).
static void
calculateConstantBufferLayout(CodeGenModule &CGM, llvm::Type *BufferTy,
                              SmallVector<unsigned> &Layout,
                              const HLSLBufferDecl *BufferDecl = nullptr) {
  assert(isa<llvm::StructType>(BufferTy) && "expected struct");
  assert(BufferTy->getStructNumElements() != 0 &&
         "empty constant buffer should not be created");

  if (BufferDecl && hasPackoffset(BufferDecl)) {
    createConstantBufferLayoutFromPackoffset(CGM, BufferTy, Layout, BufferDecl);
    return;
  }

  unsigned CBufOffset = 0, Size = 0;
  Layout.push_back(UINT_MAX); // reserve first spot for the buffer size

  for (const llvm::Type *ElTy : cast<llvm::StructType>(BufferTy)->elements()) {
    assert(!ElTy->isArrayTy() && !ElTy->isStructTy() &&
           "arrays and structs in cbuffer are not yet implemened");

    // scalar type, vector or matrix
    unsigned FieldSize = ElTy->getScalarSizeInBits() / 8;
    if (ElTy->isVectorTy())
      FieldSize *= cast<llvm::FixedVectorType>(ElTy)->getNumElements();
    assert(FieldSize <= 16 && "field side larger than constant buffer row");

    // align to the size of the field
    CBufOffset = llvm::alignTo(CBufOffset, FieldSize);
    Layout.push_back(CBufOffset);
    CBufOffset += FieldSize;
    Size = CBufOffset;
  }
  Layout[0] = Size;
}

llvm::Type *
DirectXTargetCodeGenInfo::getHLSLType(CodeGenModule &CGM, const Type *Ty,
                                      const HLSLBufferDecl *BufDecl) const {
  auto *ResType = dyn_cast<HLSLAttributedResourceType>(Ty);
  if (!ResType)
    return nullptr;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  const HLSLAttributedResourceType::Attributes &ResAttrs = ResType->getAttrs();
  switch (ResAttrs.ResourceClass) {
  case llvm::dxil::ResourceClass::UAV:
  case llvm::dxil::ResourceClass::SRV: {
    // TypedBuffer and RawBuffer both need element type
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull())
      return nullptr;

    // convert element type
    llvm::Type *ElemType = CGM.getTypes().ConvertType(ContainedTy);

    llvm::StringRef TypeName =
        ResAttrs.RawBuffer ? "dx.RawBuffer" : "dx.TypedBuffer";
    SmallVector<unsigned, 3> Ints = {/*IsWriteable*/ ResAttrs.ResourceClass ==
                                         llvm::dxil::ResourceClass::UAV,
                                     /*IsROV*/ ResAttrs.IsROV};
    if (!ResAttrs.RawBuffer)
      Ints.push_back(/*IsSigned*/ ContainedTy->isSignedIntegerType());

    return llvm::TargetExtType::get(Ctx, TypeName, {ElemType}, Ints);
  }
  case llvm::dxil::ResourceClass::CBuffer: {
    QualType StructTy = ResType->getContainedType();
    if (StructTy.isNull())
      return nullptr;

    llvm::Type *Ty = CGM.getTypes().ConvertType(StructTy);

    SmallVector<unsigned> BufferLayout;
    calculateConstantBufferLayout(CGM, Ty, BufferLayout, BufDecl);
    return llvm::TargetExtType::get(Ctx, "dx.CBuffer", {Ty}, BufferLayout);
  }
  case llvm::dxil::ResourceClass::Sampler:
    llvm_unreachable("dx.Sampler handles are not implemented yet");
    break;
  }
  llvm_unreachable("Unknown llvm::dxil::ResourceClass enum");
}

} // namespace

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createDirectXTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<DirectXTargetCodeGenInfo>(CGM.getTypes());
}
