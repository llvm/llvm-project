//===- AddDebugFoundation.cpp -- add basic debug linetable info -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass populates some debug information for the module and functions.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace fir {
#define GEN_PASS_DEF_ADDDEBUGFOUNDATION
#define GEN_PASS_DECL_ADDDEBUGFOUNDATION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-add-debug-foundation"

namespace {

class AddDebugFoundationPass
    : public fir::impl::AddDebugFoundationBase<AddDebugFoundationPass> {
public:
  void runOnOperation() override;
};

} // namespace

void AddDebugFoundationPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  std::string inputFilePath("-");
  if (auto fileLoc = module.getLoc().dyn_cast<mlir::FileLineColLoc>())
    inputFilePath = fileLoc.getFilename().getValue();

  auto getFileAttr = [context](llvm::StringRef path) -> mlir::LLVM::DIFileAttr {
    return mlir::LLVM::DIFileAttr::get(context, llvm::sys::path::filename(path),
                                       llvm::sys::path::parent_path(path));
  };

  mlir::LLVM::DIFileAttr fileAttr = getFileAttr(inputFilePath);
  mlir::StringAttr producer = mlir::StringAttr::get(context, "Flang");
  mlir::LLVM::DICompileUnitAttr cuAttr = mlir::LLVM::DICompileUnitAttr::get(
      context, mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
      llvm::dwarf::getLanguage("DW_LANG_Fortran95"), fileAttr, producer,
      /*isOptimized=*/false, mlir::LLVM::DIEmissionKind::LineTablesOnly);

  module.walk([&](mlir::func::FuncOp funcOp) {
    mlir::Location l = funcOp->getLoc();
    // If fused location has already been created then nothing to do
    // Otherwise, create a fused location.
    if (l.dyn_cast<mlir::FusedLoc>())
      return;

    llvm::StringRef funcFilePath;
    if (l.dyn_cast<mlir::FileLineColLoc>())
      funcFilePath =
          l.dyn_cast<mlir::FileLineColLoc>().getFilename().getValue();
    else
      funcFilePath = inputFilePath;

    mlir::StringAttr funcName =
        mlir::StringAttr::get(context, funcOp.getName());
    mlir::LLVM::DIBasicTypeAttr bT = mlir::LLVM::DIBasicTypeAttr::get(
        context, llvm::dwarf::DW_TAG_base_type, "void", /*sizeInBits=*/0,
        /*encoding=*/1);
    mlir::LLVM::DISubroutineTypeAttr subTypeAttr =
        mlir::LLVM::DISubroutineTypeAttr::get(
            context, llvm::dwarf::getCallingConvention("DW_CC_normal"),
            {bT, bT});
    mlir::LLVM::DIFileAttr funcFileAttr = getFileAttr(funcFilePath);

    // Only definitions need a distinct identifier and a compilation unit.
    mlir::DistinctAttr id;
    mlir::LLVM::DICompileUnitAttr compilationUnit;
    if (!funcOp.isExternal()) {
      id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      compilationUnit = cuAttr;
    }
    mlir::LLVM::DISubprogramAttr spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, fileAttr, funcName, funcName,
        funcFileAttr,
        /*line=*/1,
        /*scopeline=*/1, mlir::LLVM::DISubprogramFlags::Definition,
        subTypeAttr);
    funcOp->setLoc(builder.getFusedLoc({funcOp->getLoc()}, spAttr));
  });
}

std::unique_ptr<mlir::Pass> fir::createAddDebugFoundationPass() {
  return std::make_unique<AddDebugFoundationPass>();
}
