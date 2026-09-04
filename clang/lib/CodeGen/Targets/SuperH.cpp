//===- SuperH.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// SuperH ABI Implementation. Documented at
// https://llvm-gcc-renesas.com/manuals/SH-ABI-Specification.html
//===----------------------------------------------------------------------===//

namespace {
class SuperHABIInfo : public DefaultABIInfo {
public:
  SuperHABIInfo(CodeGenTypes &CGT)
      : DefaultABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType Ty, bool &LargeRet) const {

    // We have a total of 64 bits of return space in R0 and R1,
    // if we can fit a struct in there, do.
    if (isAggregateTypeForABI(Ty) && 
        getContext().getTypeSize(Ty) <= 64)
      return ABIArgInfo::getDirect();

    // Otherwise, we store a pointer to the struct in R2 and return that.
    if (getContext().getTypeSize(Ty) > 64) {
      LargeRet = true;
      return getNaturalAlignIndirect(Ty, getDataLayout().getAllocaAddrSpace());
    }

    // Otherwise we follow the default way which is compatible.
    return DefaultABIInfo::classifyReturnType(Ty);
  }

  ABIArgInfo classifyArgumentType(QualType Ty, unsigned &NumRegs) const {
    unsigned TySize = getContext().getTypeSize(Ty);

    // Smaller types are required to be sign extended.
    if (TySize < 32 && NumRegs > 0 && Ty->isIntegralOrEnumerationType()) {
      NumRegs -= 1;
      return ABIArgInfo::getExtend(Ty);
    }

    // If there's enough space, pass in registers.
    if (TySize <= NumRegs * 32) {
      NumRegs -= TySize / 32;
      return ABIArgInfo::getDirect();
    }

    // A pointer will be created and passsed in R4 instead, as the arguments
    // from here on out doesn't fit in the remaining registers.
    NumRegs = 0;
    return ABIArgInfo::getDirect();
  }

  void computeInfo(CGFunctionInfo &FI) const override {

    // Decide the return type.
    bool LargeRet = false;
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), LargeRet);
    
    // Decide each argument type.
    unsigned NumRegs = 4;
    for (auto &I : FI.arguments())
      I.info = classifyArgumentType(I.type, NumRegs);
  }

};

class SuperHTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  SuperHTargetCodeGenInfo(CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<SuperHABIInfo>(CGT)) {}

};
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createSuperHTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<SuperHTargetCodeGenInfo>(CGM.getTypes());
}
