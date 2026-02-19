//===- DirectX.cpp---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "CodeGenModule.h"
#include "HLSLBufferLayoutBuilder.h"
#include "TargetInfo.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

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

  llvm::Type *getHLSLType(CodeGenModule &CGM, const Type *T,
                          const CGHLSLOffsetInfo &OffsetInfo) const override;

  llvm::Type *getHLSLPadding(CodeGenModule &CGM,
                             CharUnits NumBytes) const override {
    unsigned Size = NumBytes.getQuantity();
    return llvm::TargetExtType::get(CGM.getLLVMContext(), "dx.Padding", {},
                                    {Size});
  }

  bool isHLSLPadding(llvm::Type *Ty) const override {
    if (auto *TET = dyn_cast<llvm::TargetExtType>(Ty))
      return TET->getName() == "dx.Padding";
    return false;
  }
};

llvm::Type *DirectXTargetCodeGenInfo::getHLSLType(
    CodeGenModule &CGM, const Type *Ty,
    const CGHLSLOffsetInfo &OffsetInfo) const {
  auto *ResType = dyn_cast<HLSLAttributedResourceType>(Ty);
  if (!ResType)
    return nullptr;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  const HLSLAttributedResourceType::Attributes &ResAttrs = ResType->getAttrs();
  switch (ResAttrs.ResourceClass) {
  case llvm::dxil::ResourceClass::UAV:
  case llvm::dxil::ResourceClass::SRV: {
    // TypedBuffer, RawBuffer and Texture all need element type
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull())
      return nullptr;

    // convert element type
    llvm::Type *ElemType = CGM.getTypes().ConvertTypeForMem(ContainedTy);

    bool IsRawBuffer = ResAttrs.RawBuffer;
    bool IsTexture =
        ResAttrs.ResourceDimension != llvm::dxil::ResourceDimension::Unknown;
    assert((!IsRawBuffer || !IsTexture) && "A resource cannot be both a raw "
                                           "buffer and a texture.");
    llvm::StringRef TypeName = "dx.TypedBuffer";
    if (IsRawBuffer)
      TypeName = "dx.RawBuffer";
    else if (IsTexture)
      TypeName = "dx.Texture";

    SmallVector<unsigned, 4> Ints = {/*IsWriteable*/ ResAttrs.ResourceClass ==
                                         llvm::dxil::ResourceClass::UAV,
                                     /*IsROV*/ ResAttrs.IsROV};
    if (!IsRawBuffer) {
      const clang::Type *ElemType = ContainedTy->getUnqualifiedDesugaredType();
      if (ElemType->isVectorType())
        ElemType = cast<clang::VectorType>(ElemType)
                       ->getElementType()
                       ->getUnqualifiedDesugaredType();
      Ints.push_back(/*IsSigned*/ ElemType->isSignedIntegerType());
    }

    if (IsTexture) {
      // Map ResourceDimension to dxil::ResourceKind
      llvm::dxil::ResourceKind RK = llvm::dxil::ResourceKind::Invalid;
      switch (ResAttrs.ResourceDimension) {
      case llvm::dxil::ResourceDimension::Dim1D:
        RK = llvm::dxil::ResourceKind::Texture1D;
        break;
      case llvm::dxil::ResourceDimension::Dim2D:
        RK = llvm::dxil::ResourceKind::Texture2D;
        break;
      case llvm::dxil::ResourceDimension::Dim3D:
        RK = llvm::dxil::ResourceKind::Texture3D;
        break;
      case llvm::dxil::ResourceDimension::Cube:
        RK = llvm::dxil::ResourceKind::TextureCube;
        break;
      default:
        llvm_unreachable("Unsupported resource dimension for textur.");
      }
      Ints.push_back(static_cast<unsigned>(RK));
    }

    return llvm::TargetExtType::get(Ctx, TypeName, {ElemType}, Ints);
  }
  case llvm::dxil::ResourceClass::CBuffer: {
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull() || !ContainedTy->isStructureType())
      return nullptr;

    llvm::StructType *BufferLayoutTy =
        HLSLBufferLayoutBuilder(CGM).layOutStruct(
            ContainedTy->getAsCanonical<RecordType>(), OffsetInfo);
    if (!BufferLayoutTy)
      return nullptr;

    return llvm::TargetExtType::get(Ctx, "dx.CBuffer", {BufferLayoutTy});
  }
  case llvm::dxil::ResourceClass::Sampler:
    return llvm::TargetExtType::get(Ctx, "dx.Sampler", {}, {0});
  }
  llvm_unreachable("Unknown llvm::dxil::ResourceClass enum");
}

} // namespace

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createDirectXTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<DirectXTargetCodeGenInfo>(CGM.getTypes());
}
