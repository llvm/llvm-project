//===- CIRGenAMDGPU.cpp - AMDGPU-specific logic for CIR generation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with AMDGPU-specific logic of CIR generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "clang/Basic/TargetOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/TargetParser/Triple.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenModule::emitAMDGPUMetadata() {
  // Emit code object version module flag.
  if (target.getTargetOpts().CodeObjectVersion !=
      llvm::CodeObjectVersionKind::COV_None) {
    theModule->setAttr(
        cir::CIRDialect::getAMDGPUCodeObjectVersionAttrName(),
        builder.getI32IntegerAttr(target.getTargetOpts().CodeObjectVersion));
  }

  // Emit printf kind module flag for HIP.
  if (langOpts.HIP) {
    llvm::StringRef printfKind =
        target.getTargetOpts().AMDGPUPrintfKindVal ==
                TargetOptions::AMDGPUPrintfKind::Hostcall
            ? "hostcall"
            : "buffered";
    theModule->setAttr(cir::CIRDialect::getAMDGPUPrintfKindAttrName(),
                       builder.getStringAttr(printfKind));
  }
}
