//===- DirectX.cpp---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
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

  llvm::Type *getHLSLType(CodeGenModule &CGM, const Type *T) const override;
};

llvm::Type *DirectXTargetCodeGenInfo::getHLSLType(CodeGenModule &CGM,
                                                  const Type *Ty) const {
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

    llvm::Type *ElemType = CGM.getTypes().ConvertType(ContainedTy);

    // target ext type parameters may not be named struct types
    // so we should convert any named struct types to anonymous
    // struct types in the parameter list
    llvm::Type *ConvertedElemType = ElemType;
    if (auto *STy = dyn_cast<llvm::StructType>(ElemType)) {
      if (STy->hasName())
        ConvertedElemType =
            llvm::StructType::get(Ctx, STy->elements(), /*isPacked=*/false);
    }

    llvm::StringRef TypeName =
        ResAttrs.RawBuffer ? "dx.RawBuffer" : "dx.TypedBuffer";
    SmallVector<unsigned, 3> Ints = {/*IsWriteable*/ ResAttrs.ResourceClass ==
                                         llvm::dxil::ResourceClass::UAV,
                                     /*IsROV*/ ResAttrs.IsROV};
    if (!ResAttrs.RawBuffer)
      Ints.push_back(/*IsSigned*/ ContainedTy->isSignedIntegerType());

    return llvm::TargetExtType::get(Ctx, TypeName, {ConvertedElemType}, Ints);
  }
  case llvm::dxil::ResourceClass::CBuffer:
    llvm_unreachable("dx.CBuffer handles are not implemented yet");
    break;
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
