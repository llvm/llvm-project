//===- LowerToLLVMOffloadPrintf.cpp - Lower cir.offload.printf ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers cir.offload.printf to the printf runtime interface of each
// device target.
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVMOffloadPrintf.h"
#include "LowerToLLVM.h"
#include "TargetLoweringInfo.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/TargetParser/Triple.h"

namespace cir {
namespace direct {

mlir::LogicalResult CIRToLLVMOffloadPrintfOpLowering::matchAndRewrite(
    cir::OffloadPrintfOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  llvm::StringRef tripleStr;
  if (auto tripleAttr = mod->getAttrOfType<mlir::StringAttr>(
          cir::CIRDialect::getTripleAttrName()))
    tripleStr = tripleAttr.getValue();
  llvm::Triple triple(tripleStr);

  if (triple.isNVPTX())
    return lowerNVPTXPrintf(op, adaptor.getFormat(), adaptor.getArgs(),
                            rewriter, dataLayout, symbolTables);

  if (triple.isAMDGCN() ||
      (triple.isSPIRV() && triple.getVendor() == llvm::Triple::AMD)) {
    bool isBuffered = false;
    if (auto kind = mod->getAttrOfType<mlir::StringAttr>(
            cir::CIRDialect::getAMDGPUPrintfKindAttrName()))
      isBuffered = kind.getValue() == "buffered";
    std::unique_ptr<cir::TargetLoweringInfo> targetInfo =
        triple.isSPIRV() ? cir::createSPIRVTargetLoweringInfo()
                         : cir::createAMDGPUTargetLoweringInfo();
    unsigned genericAddrSpace = targetInfo->getTargetAddrSpaceFromCIRAddrSpace(
        cir::LangAddressSpace::OffloadGeneric);
    return lowerAMDGPUPrintf(op, adaptor.getOperands(), rewriter, dataLayout,
                             symbolTables, genericAddrSpace, isBuffered);
  }

  return op.emitError() << "offload printf lowering is NYI for target '"
                        << tripleStr << "'";
}

} // namespace direct
} // namespace cir
