//===-- Tools/TargetSetup.h ------------------------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_TARGET_SETUP_H
#define FORTRAN_TOOLS_TARGET_SETUP_H

#include "flang/Common/float128.h"
#include "flang/Evaluate/target.h"
#include "flang/Frontend/TargetOptions.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include <cfloat>

namespace Fortran::tools {

[[maybe_unused]] inline static void setUpTargetCharacteristics(
    Fortran::evaluate::TargetCharacteristics &targetCharacteristics,
    const llvm::TargetMachine &targetMachine,
    const Fortran::frontend::TargetOptions &targetOptions,
    const std::string &compilerVersion, const std::string &compilerOptions) {

  const llvm::Triple &targetTriple{targetMachine.getTargetTriple()};

  if (targetTriple.getArch() == llvm::Triple::ArchType::x86_64) {
    targetCharacteristics.set_hasSubnormalFlushingControl(/*kind=*/3);
    targetCharacteristics.set_hasSubnormalFlushingControl(/*kind=*/4);
    targetCharacteristics.set_hasSubnormalFlushingControl(/*kind=*/8);
    // ieee_denorm exception support is nonstandard.
    targetCharacteristics.set_hasSubnormalExceptionSupport(/*kind=*/3);
    targetCharacteristics.set_hasSubnormalExceptionSupport(/*kind=*/4);
    targetCharacteristics.set_hasSubnormalExceptionSupport(/*kind=*/8);
  }

  if (targetTriple.isARM() || targetTriple.isAArch64()) {
    targetCharacteristics.set_haltingSupportIsUnknownAtCompileTime();
    targetCharacteristics.set_ieeeFeature(
        evaluate::IeeeFeature::Halting, false);
    targetCharacteristics.set_ieeeFeature(
        evaluate::IeeeFeature::Standard, false);
    targetCharacteristics.set_hasSubnormalFlushingControl(/*kind=*/3);
    targetCharacteristics.set_hasSubnormalFlushingControl(/*kind=*/4);
    targetCharacteristics.set_hasSubnormalFlushingControl(/*kind=*/8);
  }

  switch (targetTriple.getArch()) {
  case llvm::Triple::ArchType::amdgcn:
  case llvm::Triple::ArchType::x86_64:
    break;
  default:
    targetCharacteristics.DisableType(
        Fortran::common::TypeCategory::Real, /*kind=*/10);
    targetCharacteristics.DisableType(
        Fortran::common::TypeCategory::Complex, /*kind=*/10);
    break;
  }

  bool f128Support = false;
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> dummyModule =
      std::make_unique<llvm::Module>("quad-test", ctx);
  dummyModule->setTargetTriple(targetMachine.getTargetTriple());
  dummyModule->setDataLayout(targetMachine.createDataLayout());

  llvm::FunctionType *dummyFTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false);
  llvm::Function *dummyF = llvm::Function::Create(dummyFTy,
      llvm::GlobalValue::ExternalLinkage, "quad-test", dummyModule.get());

  const llvm::TargetLowering *dummyTLI =
      targetMachine.getSubtargetImpl(*dummyF)->getTargetLowering();

  if (dummyTLI) {
    llvm::EVT fp128EVT = llvm::EVT::getEVT(llvm::Type::getFP128Ty(ctx));

    // Query for fp128 backend support. Based on this, determine whether
    // compilation is possible on the frontend.
    bool isLegal = dummyTLI->isTypeLegal(fp128EVT);

    // We might also be able to determine fp128 backend support based on the
    // LegalizeAction value. This is likely when the value is "Legal" or
    // "LibCall". See
    // https://llvm.org/doxygen/TargetLowering_8h_source.html#l00202.
    llvm::TargetLowering::LegalizeAction LA =
        dummyTLI->getOperationAction(llvm::ISD::FADD, fp128EVT);

    // We might also be able to determine fp128 backend support based on the
    // LegalizeTypeAction value. This is likely when the value is "TypeLegal".
    // See https://llvm.org/doxygen/TargetLowering_8h_source.html#l00212.
    llvm::TargetLowering::LegalizeTypeAction LTA =
        dummyTLI->getTypeConversion(ctx, llvm::MVT::f128).first;

    if (isLegal &&
        (LA == llvm::TargetLowering::Legal ||
            LA == llvm::TargetLowering::LibCall) &&
        (LTA == llvm::TargetLowering::TypeLegal)) {
      f128Support = true;
    }
  }

  if (!f128Support) {
    targetCharacteristics.DisableType(Fortran::common::TypeCategory::Real, 16);
    targetCharacteristics.DisableType(
        Fortran::common::TypeCategory::Complex, 16);
  }

  for (auto realKind : targetOptions.disabledRealKinds) {
    targetCharacteristics.DisableType(common::TypeCategory::Real, realKind);
    targetCharacteristics.DisableType(common::TypeCategory::Complex, realKind);
  }

  for (auto intKind : targetOptions.disabledIntegerKinds)
    targetCharacteristics.DisableType(common::TypeCategory::Integer, intKind);

  targetCharacteristics.set_compilerOptionsString(compilerOptions)
      .set_compilerVersionString(compilerVersion);

  if (targetTriple.isPPC())
    targetCharacteristics.set_isPPC(true);

  if (targetTriple.isSPARC())
    targetCharacteristics.set_isSPARC(true);

  if (targetTriple.isOSWindows())
    targetCharacteristics.set_isOSWindows(true);

  // Currently the integer kind happens to be the same as the byte size
  targetCharacteristics.set_integerKindForPointer(
      targetTriple.getArchPointerBitWidth() / 8);

  // TODO: use target machine data layout to set-up the target characteristics
  // type size and alignment info.
}

} // namespace Fortran::tools

#endif // FORTRAN_TOOLS_TARGET_SETUP_H
