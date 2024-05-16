//===-------------- AddDebugInfo.cpp -- add debug info -------------------===//
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

#include "DebugTypeGenerator.h"
#include "flang/Common/Version.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace fir {
#define GEN_PASS_DEF_ADDDEBUGINFO
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-add-debug-info"

namespace {

class AddDebugInfoPass : public fir::impl::AddDebugInfoBase<AddDebugInfoPass> {
  void handleDeclareOp(fir::cg::XDeclareOp declOp,
                       mlir::LLVM::DIFileAttr fileAttr,
                       mlir::LLVM::DIScopeAttr scopeAttr,
                       fir::DebugTypeGenerator &typeGen);

public:
  AddDebugInfoPass(fir::AddDebugInfoOptions options) : Base(options) {}
  void runOnOperation() override;
};

static uint32_t getLineFromLoc(mlir::Location loc) {
  uint32_t line = 1;
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    line = fileLoc.getLine();
  return line;
}

} // namespace

void AddDebugInfoPass::handleDeclareOp(fir::cg::XDeclareOp declOp,
                                       mlir::LLVM::DIFileAttr fileAttr,
                                       mlir::LLVM::DIScopeAttr scopeAttr,
                                       fir::DebugTypeGenerator &typeGen) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  // Only accept local variables.
  if (result.second.procs.empty())
    return;

  // FIXME: There may be cases where an argument is processed a bit before
  // DeclareOp is generated. In that case, DeclareOp may point to an
  // intermediate op and not to BlockArgument. We need to find those cases and
  // walk the chain to get to the actual argument.

  unsigned argNo = 0;
  if (auto Arg = llvm::dyn_cast<mlir::BlockArgument>(declOp.getMemref()))
    argNo = Arg.getArgNumber() + 1;

  auto tyAttr = typeGen.convertType(fir::unwrapRefType(declOp.getType()),
                                    fileAttr, scopeAttr, declOp.getLoc());

  auto localVarAttr = mlir::LLVM::DILocalVariableAttr::get(
      context, scopeAttr, mlir::StringAttr::get(context, result.second.name),
      fileAttr, getLineFromLoc(declOp.getLoc()), argNo, /* alignInBits*/ 0,
      tyAttr);
  declOp->setLoc(builder.getFusedLoc({declOp->getLoc()}, localVarAttr));
}

void AddDebugInfoPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  llvm::StringRef fileName;
  std::string filePath;
  // We need 2 type of file paths here.
  // 1. Name of the file as was presented to compiler. This can be absolute
  // or relative to 2.
  // 2. Current working directory
  //
  // We are also dealing with 2 different situations below. One is normal
  // compilation where we will have a value in 'inputFilename' and we can
  // obtain the current directory using 'current_path'.
  // The 2nd case is when this pass is invoked directly from 'fir-opt' tool.
  // In that case, 'inputFilename' may be empty. Location embedded in the
  // module will be used to get file name and its directory.
  if (inputFilename.empty()) {
    if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(module.getLoc())) {
      fileName = llvm::sys::path::filename(fileLoc.getFilename().getValue());
      filePath = llvm::sys::path::parent_path(fileLoc.getFilename().getValue());
    } else
      fileName = "-";
  } else {
    fileName = inputFilename;
    llvm::SmallString<256> cwd;
    if (!llvm::sys::fs::current_path(cwd))
      filePath = cwd.str();
  }

  mlir::LLVM::DIFileAttr fileAttr =
      mlir::LLVM::DIFileAttr::get(context, fileName, filePath);
  mlir::StringAttr producer =
      mlir::StringAttr::get(context, Fortran::common::getFlangFullVersion());
  mlir::LLVM::DICompileUnitAttr cuAttr = mlir::LLVM::DICompileUnitAttr::get(
      mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
      llvm::dwarf::getLanguage("DW_LANG_Fortran95"), fileAttr, producer,
      isOptimized, debugLevel);

  module.walk([&](mlir::func::FuncOp funcOp) {
    mlir::Location l = funcOp->getLoc();
    // If fused location has already been created then nothing to do
    // Otherwise, create a fused location.
    if (mlir::dyn_cast<mlir::FusedLoc>(l))
      return;

    unsigned int CC = (funcOp.getName() == fir::NameUniquer::doProgramEntry())
                          ? llvm::dwarf::getCallingConvention("DW_CC_program")
                          : llvm::dwarf::getCallingConvention("DW_CC_normal");

    if (auto funcLoc = mlir::dyn_cast<mlir::FileLineColLoc>(l)) {
      fileName = llvm::sys::path::filename(funcLoc.getFilename().getValue());
      filePath = llvm::sys::path::parent_path(funcLoc.getFilename().getValue());
    }

    mlir::StringAttr fullName =
        mlir::StringAttr::get(context, funcOp.getName());
    auto result = fir::NameUniquer::deconstruct(funcOp.getName());
    mlir::StringAttr funcName =
        mlir::StringAttr::get(context, result.second.name);

    llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
    fir::DebugTypeGenerator typeGen(module);
    for (auto resTy : funcOp.getResultTypes()) {
      auto tyAttr =
          typeGen.convertType(resTy, fileAttr, cuAttr, funcOp.getLoc());
      types.push_back(tyAttr);
    }
    for (auto inTy : funcOp.getArgumentTypes()) {
      auto tyAttr = typeGen.convertType(fir::unwrapRefType(inTy), fileAttr,
                                        cuAttr, funcOp.getLoc());
      types.push_back(tyAttr);
    }

    mlir::LLVM::DISubroutineTypeAttr subTypeAttr =
        mlir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
    mlir::LLVM::DIFileAttr funcFileAttr =
        mlir::LLVM::DIFileAttr::get(context, fileName, filePath);

    // Only definitions need a distinct identifier and a compilation unit.
    mlir::DistinctAttr id;
    mlir::LLVM::DICompileUnitAttr compilationUnit;
    mlir::LLVM::DISubprogramFlags subprogramFlags =
        mlir::LLVM::DISubprogramFlags{};
    if (isOptimized)
      subprogramFlags = mlir::LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      compilationUnit = cuAttr;
      subprogramFlags =
          subprogramFlags | mlir::LLVM::DISubprogramFlags::Definition;
    }
    unsigned line = getLineFromLoc(l);
    auto spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, fileAttr, funcName, fullName,
        funcFileAttr, line, line, subprogramFlags, subTypeAttr);
    funcOp->setLoc(builder.getFusedLoc({funcOp->getLoc()}, spAttr));

    // Don't process variables if user asked for line tables only.
    if (debugLevel == mlir::LLVM::DIEmissionKind::LineTablesOnly)
      return;

    funcOp.walk([&](fir::cg::XDeclareOp declOp) {
      handleDeclareOp(declOp, fileAttr, spAttr, typeGen);
    });
  });
}

std::unique_ptr<mlir::Pass>
fir::createAddDebugInfoPass(fir::AddDebugInfoOptions options) {
  return std::make_unique<AddDebugInfoPass>(options);
}
