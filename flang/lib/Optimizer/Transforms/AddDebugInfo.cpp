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
                       fir::DebugTypeGenerator &typeGen,
                       mlir::SymbolTable *symbolTable);

public:
  AddDebugInfoPass(fir::AddDebugInfoOptions options) : Base(options) {}
  void runOnOperation() override;

private:
  llvm::StringMap<mlir::LLVM::DIModuleAttr> moduleMap;

  mlir::LLVM::DIModuleAttr getOrCreateModuleAttr(
      const std::string &name, mlir::LLVM::DIFileAttr fileAttr,
      mlir::LLVM::DIScopeAttr scope, unsigned line, bool decl);

  void handleGlobalOp(fir::GlobalOp glocalOp, mlir::LLVM::DIFileAttr fileAttr,
                      mlir::LLVM::DIScopeAttr scope,
                      fir::DebugTypeGenerator &typeGen,
                      mlir::SymbolTable *symbolTable,
                      fir::cg::XDeclareOp declOp);
  void handleFuncOp(mlir::func::FuncOp funcOp, mlir::LLVM::DIFileAttr fileAttr,
                    mlir::LLVM::DICompileUnitAttr cuAttr,
                    fir::DebugTypeGenerator &typeGen,
                    mlir::SymbolTable *symbolTable);
};

bool debugInfoIsAlreadySet(mlir::Location loc) {
  if (mlir::isa<mlir::FusedLoc>(loc)) {
    if (loc->findInstanceOf<mlir::FusedLocWith<fir::LocationKindAttr>>())
      return false;
    return true;
  }
  return false;
}

} // namespace

void AddDebugInfoPass::handleDeclareOp(fir::cg::XDeclareOp declOp,
                                       mlir::LLVM::DIFileAttr fileAttr,
                                       mlir::LLVM::DIScopeAttr scopeAttr,
                                       fir::DebugTypeGenerator &typeGen,
                                       mlir::SymbolTable *symbolTable) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;
  // If this DeclareOp actually represents a global then treat it as such.
  if (auto global = symbolTable->lookup<fir::GlobalOp>(declOp.getUniqName())) {
    handleGlobalOp(global, fileAttr, scopeAttr, typeGen, symbolTable, declOp);
    return;
  }

  // Only accept local variables.
  if (result.second.procs.empty())
    return;

  // FIXME: There may be cases where an argument is processed a bit before
  // DeclareOp is generated. In that case, DeclareOp may point to an
  // intermediate op and not to BlockArgument.
  // Moreover, with MLIR inlining we cannot use the BlockArgument
  // position to identify the original number of the dummy argument.
  // If we want to keep running AddDebugInfoPass late, the dummy argument
  // position in the argument list has to be expressed in FIR (e.g. as a
  // constant attribute of [hl]fir.declare/fircg.ext_declare operation that has
  // a dummy_scope operand).
  unsigned argNo = 0;
  if (fir::isDummyArgument(declOp.getMemref())) {
    auto arg = llvm::cast<mlir::BlockArgument>(declOp.getMemref());
    argNo = arg.getArgNumber() + 1;
  }

  auto tyAttr = typeGen.convertType(fir::unwrapRefType(declOp.getType()),
                                    fileAttr, scopeAttr, declOp);

  auto localVarAttr = mlir::LLVM::DILocalVariableAttr::get(
      context, scopeAttr, mlir::StringAttr::get(context, result.second.name),
      fileAttr, getLineFromLoc(declOp.getLoc()), argNo, /* alignInBits*/ 0,
      tyAttr, mlir::LLVM::DIFlags::Zero);
  declOp->setLoc(builder.getFusedLoc({declOp->getLoc()}, localVarAttr));
}

// The `module` does not have a first class representation in the `FIR`. We
// extract information about it from the name of the identifiers and keep a
// map to avoid duplication.
mlir::LLVM::DIModuleAttr AddDebugInfoPass::getOrCreateModuleAttr(
    const std::string &name, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, unsigned line, bool decl) {
  mlir::MLIRContext *context = &getContext();
  mlir::LLVM::DIModuleAttr modAttr;
  if (auto iter{moduleMap.find(name)}; iter != moduleMap.end()) {
    modAttr = iter->getValue();
  } else {
    modAttr = mlir::LLVM::DIModuleAttr::get(
        context, fileAttr, scope, mlir::StringAttr::get(context, name),
        /* configMacros */ mlir::StringAttr(),
        /* includePath */ mlir::StringAttr(),
        /* apinotes */ mlir::StringAttr(), line, decl);
    moduleMap[name] = modAttr;
  }
  return modAttr;
}

void AddDebugInfoPass::handleGlobalOp(fir::GlobalOp globalOp,
                                      mlir::LLVM::DIFileAttr fileAttr,
                                      mlir::LLVM::DIScopeAttr scope,
                                      fir::DebugTypeGenerator &typeGen,
                                      mlir::SymbolTable *symbolTable,
                                      fir::cg::XDeclareOp declOp) {
  if (debugInfoIsAlreadySet(globalOp.getLoc()))
    return;
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  // Discard entries that describe a derived type. Usually start with '.c.',
  // '.dt.' or '.n.'. It would be better if result of the deconstruct had a flag
  // for such values so that we dont have to look at string values.
  if (!result.second.name.empty() && result.second.name[0] == '.')
    return;

  unsigned line = getLineFromLoc(globalOp.getLoc());

  // DWARF5 says following about the fortran modules:
  // A Fortran 90 module may also be represented by a module entry
  // (but no declaration attribute is warranted because Fortran has no concept
  // of a corresponding module body).
  // But in practice, compilers use declaration attribute with a module in cases
  // where module was defined in another source file (only being used in this
  // one). The isInitialized() seems to provide the right information
  // but inverted. It is true where module is actually defined but false where
  // it is used.
  // FIXME: Currently we don't have the line number on which a module was
  // declared. We are using a best guess of line - 1 where line is the source
  // line of the first member of the module that we encounter.

  if (result.second.procs.empty()) {
    // Only look for module if this variable is not part of a function.
    if (result.second.modules.empty())
      return;

    // Modules are generated at compile unit scope
    if (mlir::LLVM::DISubprogramAttr sp =
            mlir::dyn_cast_if_present<mlir::LLVM::DISubprogramAttr>(scope))
      scope = sp.getCompileUnit();

    scope = getOrCreateModuleAttr(result.second.modules[0], fileAttr, scope,
                                  line - 1, !globalOp.isInitialized());
  }
  mlir::LLVM::DITypeAttr diType =
      typeGen.convertType(globalOp.getType(), fileAttr, scope, declOp);
  auto gvAttr = mlir::LLVM::DIGlobalVariableAttr::get(
      context, scope, mlir::StringAttr::get(context, result.second.name),
      mlir::StringAttr::get(context, globalOp.getName()), fileAttr, line,
      diType, /*isLocalToUnit*/ false,
      /*isDefinition*/ globalOp.isInitialized(), /* alignInBits*/ 0);
  globalOp->setLoc(builder.getFusedLoc({globalOp->getLoc()}, gvAttr));
}

void AddDebugInfoPass::handleFuncOp(mlir::func::FuncOp funcOp,
                                    mlir::LLVM::DIFileAttr fileAttr,
                                    mlir::LLVM::DICompileUnitAttr cuAttr,
                                    fir::DebugTypeGenerator &typeGen,
                                    mlir::SymbolTable *symbolTable) {
  mlir::Location l = funcOp->getLoc();
  // If fused location has already been created then nothing to do
  // Otherwise, create a fused location.
  if (debugInfoIsAlreadySet(l))
    return;

  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  llvm::StringRef fileName(fileAttr.getName());
  llvm::StringRef filePath(fileAttr.getDirectory());
  unsigned int CC = (funcOp.getName() == fir::NameUniquer::doProgramEntry())
                        ? llvm::dwarf::getCallingConvention("DW_CC_program")
                        : llvm::dwarf::getCallingConvention("DW_CC_normal");

  if (auto funcLoc = mlir::dyn_cast<mlir::FileLineColLoc>(l)) {
    fileName = llvm::sys::path::filename(funcLoc.getFilename().getValue());
    filePath = llvm::sys::path::parent_path(funcLoc.getFilename().getValue());
  }

  mlir::StringAttr fullName = mlir::StringAttr::get(context, funcOp.getName());
  mlir::Attribute attr = funcOp->getAttr(fir::getInternalFuncNameAttrName());
  mlir::StringAttr funcName =
      (attr) ? mlir::cast<mlir::StringAttr>(attr)
             : mlir::StringAttr::get(context, funcOp.getName());

  auto result = fir::NameUniquer::deconstruct(funcName);
  funcName = mlir::StringAttr::get(context, result.second.name);

  llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
  for (auto resTy : funcOp.getResultTypes()) {
    auto tyAttr =
        typeGen.convertType(resTy, fileAttr, cuAttr, /*declOp=*/nullptr);
    types.push_back(tyAttr);
  }
  for (auto inTy : funcOp.getArgumentTypes()) {
    auto tyAttr = typeGen.convertType(fir::unwrapRefType(inTy), fileAttr,
                                      cuAttr, /*declOp=*/nullptr);
    types.push_back(tyAttr);
  }

  mlir::LLVM::DISubroutineTypeAttr subTypeAttr =
      mlir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
  mlir::LLVM::DIFileAttr funcFileAttr =
      mlir::LLVM::DIFileAttr::get(context, fileName, filePath);

  // Only definitions need a distinct identifier and a compilation unit.
  mlir::DistinctAttr id;
  mlir::LLVM::DIScopeAttr Scope = fileAttr;
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
  if (fir::isInternalProcedure(funcOp)) {
    // For contained functions, the scope is the parent subroutine.
    mlir::SymbolRefAttr sym = mlir::cast<mlir::SymbolRefAttr>(
        funcOp->getAttr(fir::getHostSymbolAttrName()));
    if (sym) {
      if (auto func =
              symbolTable->lookup<mlir::func::FuncOp>(sym.getLeafReference())) {
        // Make sure that parent is processed.
        handleFuncOp(func, fileAttr, cuAttr, typeGen, symbolTable);
        if (auto fusedLoc =
                mlir::dyn_cast_if_present<mlir::FusedLoc>(func.getLoc())) {
          if (auto spAttr =
                  mlir::dyn_cast_if_present<mlir::LLVM::DISubprogramAttr>(
                      fusedLoc.getMetadata()))
            Scope = spAttr;
        }
      }
    }
  } else if (!result.second.modules.empty()) {
    Scope = getOrCreateModuleAttr(result.second.modules[0], fileAttr, cuAttr,
                                  line - 1, false);
  }

  auto spAttr = mlir::LLVM::DISubprogramAttr::get(
      context, id, compilationUnit, Scope, funcName, fullName, funcFileAttr,
      line, line, subprogramFlags, subTypeAttr, /*retainedNodes=*/{});
  funcOp->setLoc(builder.getFusedLoc({funcOp->getLoc()}, spAttr));

  // Don't process variables if user asked for line tables only.
  if (debugLevel == mlir::LLVM::DIEmissionKind::LineTablesOnly)
    return;

  funcOp.walk([&](fir::cg::XDeclareOp declOp) {
    // FIXME: We currently dont handle variables that are not in the entry
    // blocks of the fuctions. These may be variable or arguments used in the
    // OpenMP target regions.
    if (&funcOp.front() == declOp->getBlock())
      handleDeclareOp(declOp, fileAttr, spAttr, typeGen, symbolTable);
  });
}

void AddDebugInfoPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTable symbolTable(module);
  llvm::StringRef fileName;
  std::string filePath;
  std::optional<mlir::DataLayout> dl =
      fir::support::getOrSetDataLayout(module, /*allowDefaultLayout=*/true);
  if (!dl) {
    mlir::emitError(module.getLoc(), "Missing data layout attribute in module");
    signalPassFailure();
    return;
  }
  fir::DebugTypeGenerator typeGen(module, &symbolTable, *dl);
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
    handleFuncOp(funcOp, fileAttr, cuAttr, typeGen, &symbolTable);
  });
  // Process any global which was not processed through DeclareOp.
  if (debugLevel == mlir::LLVM::DIEmissionKind::Full) {
    // Process 'GlobalOp' only if full debug info is requested.
    for (auto globalOp : module.getOps<fir::GlobalOp>())
      handleGlobalOp(globalOp, fileAttr, cuAttr, typeGen, &symbolTable,
                     /*declOp=*/nullptr);
  }
}

std::unique_ptr<mlir::Pass>
fir::createAddDebugInfoPass(fir::AddDebugInfoOptions options) {
  return std::make_unique<AddDebugInfoPass>(options);
}
