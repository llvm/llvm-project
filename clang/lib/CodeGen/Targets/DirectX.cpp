//===- DirectX.cpp
//-----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/ErrorHandling.h"

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
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  if (auto *BuiltinTy = dyn_cast<BuiltinType>(Ty)) {
    switch (BuiltinTy->getKind()) {
    case BuiltinType::HLSLResource: {
      // FIXME: translate __hlsl_resource_t to target("dx.TypedBuffer", i32, 1,
      // 0, 1) only for now (RWBuffer<int>); more work us needed to determine
      // the target ext type and its parameters based on the handle type
      // attributes (not yet implemented)
      llvm::IntegerType *ElemType = llvm::IntegerType::getInt32Ty(Ctx);
      ArrayRef<unsigned> Flags = {/*IsWriteable*/ 1, /*IsROV*/ 0,
                                  /*IsSigned*/ 1};
      return llvm::TargetExtType::get(Ctx, "dx.TypedBuffer", {ElemType}, Flags);
    }
    default:
      llvm_unreachable("unhandled builtin type");
    }
  }
  return nullptr;
}

} // namespace

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createDirectXTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<DirectXTargetCodeGenInfo>(CGM.getTypes());
}
