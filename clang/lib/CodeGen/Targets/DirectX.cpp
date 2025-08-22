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
#include "clang/AST/TypeBase.h"
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

  llvm::Type *
  getHLSLType(CodeGenModule &CGM, const Type *T,
              const SmallVector<int32_t> *Packoffsets = nullptr) const override;
};

llvm::Type *DirectXTargetCodeGenInfo::getHLSLType(
    CodeGenModule &CGM, const Type *Ty,
    const SmallVector<int32_t> *Packoffsets) const {
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
    llvm::Type *ElemType = CGM.getTypes().ConvertTypeForMem(ContainedTy);

    llvm::StringRef TypeName =
        ResAttrs.RawBuffer ? "dx.RawBuffer" : "dx.TypedBuffer";
    SmallVector<unsigned, 3> Ints = {/*IsWriteable*/ ResAttrs.ResourceClass ==
                                         llvm::dxil::ResourceClass::UAV,
                                     /*IsROV*/ ResAttrs.IsROV};
    if (!ResAttrs.RawBuffer) {
      const clang::Type *ElemType = ContainedTy->getUnqualifiedDesugaredType();
      if (ElemType->isVectorType())
        ElemType = cast<clang::VectorType>(ElemType)
                       ->getElementType()
                       ->getUnqualifiedDesugaredType();
      Ints.push_back(/*IsSigned*/ ElemType->isSignedIntegerType());
    }

    return llvm::TargetExtType::get(Ctx, TypeName, {ElemType}, Ints);
  }
  case llvm::dxil::ResourceClass::CBuffer: {
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull() || !ContainedTy->isStructureType())
      return nullptr;

    llvm::Type *BufferLayoutTy =
        HLSLBufferLayoutBuilder(CGM, "dx.Layout")
            .createLayoutType(ContainedTy->castAsCanonical<RecordType>(),
                              Packoffsets);
    if (!BufferLayoutTy)
      return nullptr;

    return llvm::TargetExtType::get(Ctx, "dx.CBuffer", {BufferLayoutTy});
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
