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
  auto *BuiltinTy = dyn_cast<BuiltinType>(Ty);
  if (!BuiltinTy || BuiltinTy->getKind() != BuiltinType::HLSLResource)
    return nullptr;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  // FIXME: translate __hlsl_resource_t to target("dx.TypedBuffer", <4 x float>,
  // 1, 0, 0) only for now (RWBuffer<float4>); more work us needed to determine
  // the target ext type and its parameters based on the handle type
  // attributes (not yet implemented)
  llvm::FixedVectorType *ElemType =
      llvm::FixedVectorType::get(llvm::Type::getFloatTy(Ctx), 4);
  unsigned Flags[] = {/*IsWriteable*/ 1, /*IsROV*/ 0, /*IsSigned*/ 0};
  return llvm::TargetExtType::get(Ctx, "dx.TypedBuffer", {ElemType}, Flags);
}

} // namespace

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createDirectXTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<DirectXTargetCodeGenInfo>(CGM.getTypes());
}
