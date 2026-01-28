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
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Support/Version.h"
#include "mlir/Dialect/DLTI/DLTI.h"
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
#include "llvm/Support/FormatVariadic.h"
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
                       mlir::SymbolTable *symbolTable, mlir::Value dummyScope);

public:
  AddDebugInfoPass(fir::AddDebugInfoOptions options) : Base(options) {}
  void runOnOperation() override;

private:
  llvm::StringMap<mlir::LLVM::DIModuleAttr> moduleMap;
  llvm::StringMap<mlir::LLVM::DICommonBlockAttr> commonBlockMap;
  // List of GlobalVariableExpressionAttr that are attached to a given global
  // that represents the storage for common block.
  llvm::DenseMap<fir::GlobalOp, llvm::SmallVector<mlir::Attribute>>
      globalToGlobalExprsMap;

  mlir::LLVM::DIModuleAttr getOrCreateModuleAttr(
      const std::string &name, mlir::LLVM::DIFileAttr fileAttr,
      mlir::LLVM::DIScopeAttr scope, unsigned line, bool decl);
  mlir::LLVM::DICommonBlockAttr
  getOrCreateCommonBlockAttr(llvm::StringRef name,
                             mlir::LLVM::DIFileAttr fileAttr,
                             mlir::LLVM::DIScopeAttr scope, unsigned line);

  void handleGlobalOp(fir::GlobalOp glocalOp, mlir::LLVM::DIFileAttr fileAttr,
                      mlir::LLVM::DIScopeAttr scope,
                      fir::DebugTypeGenerator &typeGen,
                      mlir::SymbolTable *symbolTable,
                      fir::cg::XDeclareOp declOp);
  void handleFuncOp(mlir::func::FuncOp funcOp, mlir::LLVM::DIFileAttr fileAttr,
                    mlir::LLVM::DICompileUnitAttr cuAttr,
                    fir::DebugTypeGenerator &typeGen,
                    mlir::SymbolTable *symbolTable);
  void handleOnlyClause(
      fir::UseStmtOp useOp, mlir::LLVM::DISubprogramAttr spAttr,
      mlir::LLVM::DIFileAttr fileAttr, mlir::SymbolTable *symbolTable,
      llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> &importedModules);
  void handleRenamesWithoutOnly(
      fir::UseStmtOp useOp, mlir::LLVM::DISubprogramAttr spAttr,
      mlir::LLVM::DIModuleAttr modAttr, mlir::LLVM::DIFileAttr fileAttr,
      mlir::SymbolTable *symbolTable,
      llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> &importedModules);
  void handleUseStatements(
      mlir::func::FuncOp funcOp, mlir::LLVM::DISubprogramAttr spAttr,
      mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DICompileUnitAttr cuAttr,
      mlir::SymbolTable *symbolTable,
      llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> &importedEntities);
  std::optional<mlir::LLVM::DIImportedEntityAttr> createImportedDeclForGlobal(
      llvm::StringRef symbolName, mlir::LLVM::DISubprogramAttr spAttr,
      mlir::LLVM::DIFileAttr fileAttr, mlir::StringAttr localNameAttr,
      mlir::SymbolTable *symbolTable);
  bool createCommonBlockGlobal(fir::cg::XDeclareOp declOp,
                               const std::string &name,
                               mlir::LLVM::DIFileAttr fileAttr,
                               mlir::LLVM::DIScopeAttr scopeAttr,
                               fir::DebugTypeGenerator &typeGen,
                               mlir::SymbolTable *symbolTable);
  std::optional<mlir::LLVM::DIModuleAttr>
  getModuleAttrFromGlobalOp(fir::GlobalOp globalOp,
                            mlir::LLVM::DIFileAttr fileAttr,
                            mlir::LLVM::DIScopeAttr scope);
};

bool debugInfoIsAlreadySet(mlir::Location loc) {
  if (mlir::isa<mlir::FusedLoc>(loc)) {
    if (loc->findInstanceOf<mlir::FusedLocWith<fir::LocationKindAttr>>())
      return false;
    return true;
  }
  return false;
}

// Generates the name for the artificial DISubprogram that we are going to
// generate for omp::TargetOp. Its logic is borrowed from
// getTargetEntryUniqueInfo and
// TargetRegionEntryInfo::getTargetRegionEntryFnName to generate the same name.
// But even if there was a slight mismatch, it is not a problem because this
// name is artificial and not important to debug experience.
mlir::StringAttr getTargetFunctionName(mlir::MLIRContext *context,
                                       mlir::Location Loc,
                                       llvm::StringRef parentName) {
  auto fileLoc = Loc->findInstanceOf<mlir::FileLineColLoc>();

  assert(fileLoc && "No file found from location");
  llvm::StringRef fileName = fileLoc.getFilename().getValue();

  llvm::sys::fs::UniqueID id;
  uint64_t line = fileLoc.getLine();
  size_t fileId;
  size_t deviceId;
  if (auto ec = llvm::sys::fs::getUniqueID(fileName, id)) {
    fileId = llvm::hash_value(fileName.str());
    deviceId = 0xdeadf17e;
  } else {
    fileId = id.getFile();
    deviceId = id.getDevice();
  }
  return mlir::StringAttr::get(
      context,
      std::string(llvm::formatv("__omp_offloading_{0:x-}_{1:x-}_{2}_l{3}",
                                deviceId, fileId, parentName, line)));
}

} // namespace

// Check if a global represents a module variable
static bool isModuleVariable(fir::GlobalOp globalOp) {
  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  return result.first == fir::NameUniquer::NameKind::VARIABLE &&
         result.second.procs.empty() && !result.second.modules.empty();
}

// Look up DIGlobalVariable from a global symbol
static std::optional<mlir::LLVM::DIGlobalVariableAttr>
lookupDIGlobalVariable(llvm::StringRef symbolName,
                       mlir::SymbolTable *symbolTable) {
  if (auto globalOp = symbolTable->lookup<fir::GlobalOp>(symbolName)) {
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(globalOp.getLoc())) {
      if (auto metadata = fusedLoc.getMetadata()) {
        if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(metadata)) {
          for (auto elem : arrayAttr) {
            if (auto gvExpr =
                    mlir::dyn_cast<mlir::LLVM::DIGlobalVariableExpressionAttr>(
                        elem))
              return gvExpr.getVar();
          }
        }
      }
    }
  }
  return std::nullopt;
}

bool AddDebugInfoPass::createCommonBlockGlobal(
    fir::cg::XDeclareOp declOp, const std::string &name,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DIScopeAttr scopeAttr,
    fir::DebugTypeGenerator &typeGen, mlir::SymbolTable *symbolTable) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  std::optional<std::int64_t> offset;
  mlir::Value storage = declOp.getStorage();
  if (!storage)
    return false;

  // Extract offset from storage_offset attribute
  uint64_t storageOffset = declOp.getStorageOffset();
  if (storageOffset != 0)
    offset = static_cast<std::int64_t>(storageOffset);

  // Get the GlobalOp from the storage value.
  // The storage may be wrapped in ConvertOp, so unwrap it first.
  mlir::Operation *storageOp = storage.getDefiningOp();
  if (auto convertOp = mlir::dyn_cast_if_present<fir::ConvertOp>(storageOp))
    storageOp = convertOp.getValue().getDefiningOp();

  auto addrOfOp = mlir::dyn_cast_if_present<fir::AddrOfOp>(storageOp);
  if (!addrOfOp)
    return false;

  mlir::SymbolRefAttr sym = addrOfOp.getSymbol();
  fir::GlobalOp global =
      symbolTable->lookup<fir::GlobalOp>(sym.getRootReference());
  if (!global)
    return false;

  // Check if the global is actually a common block by demangling its name.
  // Module EQUIVALENCE variables also use storage operands but are mangled
  // as VARIABLE type, so we reject them to avoid treating them as common
  // blocks.
  llvm::StringRef globalSymbol = sym.getRootReference();
  auto globalResult = fir::NameUniquer::deconstruct(globalSymbol);
  if (globalResult.first == fir::NameUniquer::NameKind::VARIABLE)
    return false;

  // FIXME: We are trying to extract the name of the common block from the
  // name of the global. As part of mangling, GetCommonBlockObjectName can
  // add a trailing _ in the name of that global. The demangle function
  // does not seem to handle such cases. So the following hack is used to
  // remove the trailing '_'.
  llvm::StringRef commonName = globalSymbol;
  if (commonName != Fortran::common::blankCommonObjectName &&
      !commonName.empty() && commonName.back() == '_')
    commonName = commonName.drop_back();

  // Create the debug attributes.
  unsigned line = getLineFromLoc(global.getLoc());
  mlir::LLVM::DICommonBlockAttr commonBlock =
      getOrCreateCommonBlockAttr(commonName, fileAttr, scopeAttr, line);

  mlir::LLVM::DITypeAttr diType = typeGen.convertType(
      fir::unwrapRefType(declOp.getType()), fileAttr, scopeAttr, declOp);

  line = getLineFromLoc(declOp.getLoc());
  auto gvAttr = mlir::LLVM::DIGlobalVariableAttr::get(
      context, commonBlock, mlir::StringAttr::get(context, name),
      declOp.getUniqName(), fileAttr, line, diType,
      /*isLocalToUnit*/ false, /*isDefinition*/ true, /* alignInBits*/ 0);

  // Create DIExpression for offset if needed
  mlir::LLVM::DIExpressionAttr expr;
  if (offset && *offset != 0) {
    llvm::SmallVector<mlir::LLVM::DIExpressionElemAttr> ops;
    ops.push_back(mlir::LLVM::DIExpressionElemAttr::get(
        context, llvm::dwarf::DW_OP_plus_uconst, *offset));
    expr = mlir::LLVM::DIExpressionAttr::get(context, ops);
  }

  auto dbgExpr = mlir::LLVM::DIGlobalVariableExpressionAttr::get(
      global.getContext(), gvAttr, expr);
  globalToGlobalExprsMap[global].push_back(dbgExpr);

  return true;
}

void AddDebugInfoPass::handleDeclareOp(fir::cg::XDeclareOp declOp,
                                       mlir::LLVM::DIFileAttr fileAttr,
                                       mlir::LLVM::DIScopeAttr scopeAttr,
                                       fir::DebugTypeGenerator &typeGen,
                                       mlir::SymbolTable *symbolTable,
                                       mlir::Value dummyScope) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  if (createCommonBlockGlobal(declOp, result.second.name, fileAttr, scopeAttr,
                              typeGen, symbolTable))
    return;

  // If this DeclareOp actually represents a global then treat it as such.
  mlir::Operation *defOp = declOp.getMemref().getDefiningOp();
  if (defOp && llvm::isa<fir::AddrOfOp>(defOp)) {
    if (auto global =
            symbolTable->lookup<fir::GlobalOp>(declOp.getUniqName())) {
      handleGlobalOp(global, fileAttr, scopeAttr, typeGen, symbolTable, declOp);
      return;
    }
  }

  // Get the dummy argument position from the explicit attribute.
  unsigned argNo = 0;
  if (dummyScope && declOp.getDummyScope() == dummyScope) {
    if (auto argNoOpt = declOp.getDummyArgNo())
      argNo = *argNoOpt;
  }

  auto tyAttr = typeGen.convertType(fir::unwrapRefType(declOp.getType()),
                                    fileAttr, scopeAttr, declOp);

  auto localVarAttr = mlir::LLVM::DILocalVariableAttr::get(
      context, scopeAttr, mlir::StringAttr::get(context, result.second.name),
      fileAttr, getLineFromLoc(declOp.getLoc()), argNo, /* alignInBits*/ 0,
      tyAttr, mlir::LLVM::DIFlags::Zero);
  declOp->setLoc(builder.getFusedLoc({declOp->getLoc()}, localVarAttr));
}

mlir::LLVM::DICommonBlockAttr AddDebugInfoPass::getOrCreateCommonBlockAttr(
    llvm::StringRef name, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, unsigned line) {
  mlir::MLIRContext *context = &getContext();
  mlir::LLVM::DICommonBlockAttr cbAttr;
  if (auto iter{commonBlockMap.find(name)}; iter != commonBlockMap.end()) {
    cbAttr = iter->getValue();
  } else {
    cbAttr = mlir::LLVM::DICommonBlockAttr::get(
        context, scope, nullptr, mlir::StringAttr::get(context, name), fileAttr,
        line);
    commonBlockMap[name] = cbAttr;
  }
  return cbAttr;
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
    // When decl is true, it means that module is only being used in this
    // compilation unit and it is defined elsewhere. But if the file/line/scope
    // fields are valid, the module is not merged with its definition and is
    // considered different. So we only set those fields when decl is false.
    modAttr = mlir::LLVM::DIModuleAttr::get(
        context, decl ? nullptr : fileAttr, decl ? nullptr : scope,
        mlir::StringAttr::get(context, name),
        /* configMacros */ mlir::StringAttr(),
        /* includePath */ mlir::StringAttr(),
        /* apinotes */ mlir::StringAttr(), decl ? 0 : line, decl);
    moduleMap[name] = modAttr;
  }
  return modAttr;
}

/// If globalOp represents a module variable, return a ModuleAttr that
/// represents that module.
std::optional<mlir::LLVM::DIModuleAttr>
AddDebugInfoPass::getModuleAttrFromGlobalOp(fir::GlobalOp globalOp,
                                            mlir::LLVM::DIFileAttr fileAttr,
                                            mlir::LLVM::DIScopeAttr scope) {
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  // Only look for module if this variable is not part of a function.
  if (!result.second.procs.empty() || result.second.modules.empty())
    return std::nullopt;

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
  unsigned line = getLineFromLoc(globalOp.getLoc());

  mlir::LLVM::DISubprogramAttr sp =
      mlir::dyn_cast_if_present<mlir::LLVM::DISubprogramAttr>(scope);
  // Modules are generated at compile unit scope
  if (sp)
    scope = sp.getCompileUnit();

  return getOrCreateModuleAttr(result.second.modules[0], fileAttr, scope,
                               std::max(line - 1, (unsigned)1),
                               !globalOp.isInitialized());
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

  if (fir::NameUniquer::isSpecialSymbol(result.second.name))
    return;

  unsigned line = getLineFromLoc(globalOp.getLoc());
  std::optional<mlir::LLVM::DIModuleAttr> modOpt =
      getModuleAttrFromGlobalOp(globalOp, fileAttr, scope);
  if (modOpt)
    scope = *modOpt;

  mlir::LLVM::DITypeAttr diType =
      typeGen.convertType(globalOp.getType(), fileAttr, scope, declOp);
  auto gvAttr = mlir::LLVM::DIGlobalVariableAttr::get(
      context, scope, mlir::StringAttr::get(context, result.second.name),
      mlir::StringAttr::get(context, globalOp.getName()), fileAttr, line,
      diType, /*isLocalToUnit*/ false,
      /*isDefinition*/ globalOp.isInitialized(), /* alignInBits*/ 0);
  auto dbgExpr = mlir::LLVM::DIGlobalVariableExpressionAttr::get(
      globalOp.getContext(), gvAttr, nullptr);
  auto arrayAttr = mlir::ArrayAttr::get(context, {dbgExpr});
  globalOp->setLoc(builder.getFusedLoc({globalOp.getLoc()}, arrayAttr));
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

  // try to use a better function name than _QQmain for the program statement
  bool isMain = false;
  if (funcName == fir::NameUniquer::doProgramEntry()) {
    isMain = true;
    mlir::StringAttr bindcName =
        funcOp->getAttrOfType<mlir::StringAttr>(fir::getSymbolAttrName());
    if (bindcName)
      funcName = bindcName;
  }

  llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
  for (auto resTy : funcOp.getResultTypes()) {
    auto tyAttr =
        typeGen.convertType(resTy, fileAttr, cuAttr, /*declOp=*/nullptr);
    types.push_back(tyAttr);
  }
  // If no return type then add a null type as a place holder for that.
  if (types.empty())
    types.push_back(mlir::LLVM::DINullTypeAttr::get(context));
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
  mlir::DistinctAttr id, id2;
  mlir::LLVM::DIScopeAttr Scope = fileAttr;
  mlir::LLVM::DICompileUnitAttr compilationUnit;
  mlir::LLVM::DISubprogramFlags subprogramFlags =
      mlir::LLVM::DISubprogramFlags{};
  if (isOptimized)
    subprogramFlags = mlir::LLVM::DISubprogramFlags::Optimized;
  if (isMain)
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::MainSubprogram;
  if (!funcOp.isExternal()) {
    // Place holder and final function have to have different IDs, otherwise
    // translation code will reject one of them.
    id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    id2 = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
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

  auto addTargetOpDISP = [&](bool lineTableOnly,
                             llvm::ArrayRef<mlir::LLVM::DINodeAttr> entities) {
    // When we process the DeclareOp inside the OpenMP target region, all the
    // variables get the DISubprogram of the parent function of the target op as
    // the scope. In the codegen (to llvm ir), OpenMP target op results in the
    // creation of a separate function. As the variables in the debug info have
    // the DISubprogram of the parent function as the scope, the variables
    // need to be updated at codegen time to avoid verification failures.

    // This updating after the fact becomes more and more difficult when types
    // are dependent on local variables like in the case of variable size arrays
    // or string. We not only have to generate new variables but also new types.
    // We can avoid this problem by generating a DISubprogramAttr here for the
    // target op and make sure that all the variables inside the target region
    // get the correct scope in the first place.
    funcOp.walk([&](mlir::omp::TargetOp targetOp) {
      unsigned line = getLineFromLoc(targetOp.getLoc());
      mlir::StringAttr name =
          getTargetFunctionName(context, targetOp.getLoc(), funcOp.getName());
      mlir::LLVM::DISubprogramFlags flags =
          mlir::LLVM::DISubprogramFlags::Definition |
          mlir::LLVM::DISubprogramFlags::LocalToUnit;
      if (isOptimized)
        flags = flags | mlir::LLVM::DISubprogramFlags::Optimized;

      mlir::DistinctAttr id =
          mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      llvm::SmallVector<mlir::LLVM::DITypeAttr> types;
      types.push_back(mlir::LLVM::DINullTypeAttr::get(context));
      for (auto arg : targetOp.getRegion().getArguments()) {
        auto tyAttr = typeGen.convertType(fir::unwrapRefType(arg.getType()),
                                          fileAttr, cuAttr, /*declOp=*/nullptr);
        types.push_back(tyAttr);
      }
      CC = llvm::dwarf::getCallingConvention("DW_CC_normal");
      mlir::LLVM::DISubroutineTypeAttr spTy =
          mlir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
      if (lineTableOnly || entities.empty()) {
        auto spAttr = mlir::LLVM::DISubprogramAttr::get(
            context, id, compilationUnit, Scope, name, name, funcFileAttr, line,
            line, flags, spTy, /*retainedNodes=*/{}, /*annotations=*/{});
        targetOp->setLoc(builder.getFusedLoc({targetOp.getLoc()}, spAttr));
        return;
      }
      mlir::DistinctAttr recId =
          mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      auto spAttr = mlir::LLVM::DISubprogramAttr::get(
          context, recId, /*isRecSelf=*/true, id, compilationUnit, Scope, name,
          name, funcFileAttr, line, line, flags, spTy, /*retainedNodes=*/{},
          /*annotations=*/{});

      // Make sure that information about the imported modules is copied in the
      // new function.
      llvm::SmallVector<mlir::LLVM::DINodeAttr> opEntities;
      for (mlir::LLVM::DINodeAttr N : entities) {
        if (auto entity = mlir::dyn_cast<mlir::LLVM::DIImportedEntityAttr>(N)) {
          auto importedEntity = mlir::LLVM::DIImportedEntityAttr::get(
              context, entity.getTag(), spAttr, entity.getEntity(),
              entity.getFile(), entity.getLine(), entity.getName(),
              entity.getElements());
          opEntities.push_back(importedEntity);
        }
      }

      id = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      spAttr = mlir::LLVM::DISubprogramAttr::get(
          context, recId, /*isRecSelf=*/false, id, compilationUnit, Scope, name,
          name, funcFileAttr, line, line, flags, spTy, opEntities,
          /*annotations=*/{});
      targetOp->setLoc(builder.getFusedLoc({targetOp.getLoc()}, spAttr));
    });
  };

  // Don't process variables if user asked for line tables only.
  if (debugLevel == mlir::LLVM::DIEmissionKind::LineTablesOnly) {
    auto spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, Scope, funcName, fullName, funcFileAttr,
        line, line, subprogramFlags, subTypeAttr, /*retainedNodes=*/{},
        /*annotations=*/{});
    funcOp->setLoc(builder.getFusedLoc({l}, spAttr));
    addTargetOpDISP(/*lineTableOnly=*/true, /*entities=*/{});
    return;
  }

  // Check if there are any USE statements
  bool hasUseStmts = false;
  funcOp.walk([&](fir::UseStmtOp useOp) {
    hasUseStmts = true;
    return mlir::WalkResult::interrupt();
  });

  mlir::LLVM::DISubprogramAttr spAttr;
  llvm::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;

  if (hasUseStmts) {
    mlir::DistinctAttr recId =
        mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
    // The debug attribute in MLIR are readonly once created. But in case of
    // imported entities, we have a circular dependency. The
    // DIImportedEntityAttr requires scope information (DISubprogramAttr in this
    // case) and DISubprogramAttr requires the list of imported entities. The
    // MLIR provides a way where a DISubprogramAttr an be created with a certain
    // recID and be used in places like DIImportedEntityAttr. After that another
    // DISubprogramAttr can be created with same recID but with list of entities
    // now available. The MLIR translation code takes care of updating the
    // references. Note that references will be updated only in the things that
    // are part of DISubprogramAttr (like DIImportedEntityAttr) so we have to
    // create the final DISubprogramAttr before we process local variables.
    // Look at DIRecursiveTypeAttrInterface for more details.
    spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/true, id, compilationUnit, Scope,
        funcName, fullName, funcFileAttr, line, line, subprogramFlags,
        subTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});

    // Process USE statements (module globals are already processed)
    llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> importedEntities;
    handleUseStatements(funcOp, spAttr, fileAttr, cuAttr, symbolTable,
                        importedEntities);

    retainedNodes.append(importedEntities.begin(), importedEntities.end());

    // Create final DISubprogramAttr with imported entities and same recId
    spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/false, id2, compilationUnit, Scope,
        funcName, fullName, funcFileAttr, line, line, subprogramFlags,
        subTypeAttr, retainedNodes, /*annotations=*/{});
  } else
    // No USE statements - create final DISubprogramAttr directly
    spAttr = mlir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, Scope, funcName, fullName, funcFileAttr,
        line, line, subprogramFlags, subTypeAttr, /*retainedNodes=*/{},
        /*annotations=*/{});

  funcOp->setLoc(builder.getFusedLoc({l}, spAttr));
  addTargetOpDISP(/*lineTableOnly=*/false, retainedNodes);

  // Find the first dummy_scope definition. This is the one of the current
  // function. The other ones may come from inlined calls. The variables inside
  // those inlined calls should not be identified as arguments of the current
  // function.
  mlir::Value dummyScope;
  funcOp.walk([&](fir::UndefOp undef) -> mlir::WalkResult {
    // TODO: delay fir.dummy_scope translation to undefined until
    // codegeneration. This is nicer and safer to match.
    if (llvm::isa<fir::DummyScopeType>(undef.getType())) {
      dummyScope = undef;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  funcOp.walk([&](fir::cg::XDeclareOp declOp) {
    mlir::LLVM::DISubprogramAttr spTy = spAttr;
    if (auto tOp = declOp->getParentOfType<mlir::omp::TargetOp>()) {
      if (auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(tOp.getLoc())) {
        if (auto sp = llvm::dyn_cast<mlir::LLVM::DISubprogramAttr>(
                fusedLoc.getMetadata()))
          spTy = sp;
      }
    }
    handleDeclareOp(declOp, fileAttr, spTy, typeGen, symbolTable, dummyScope);
  });
  // commonBlockMap ensures that we don't create multiple DICommonBlockAttr of
  // the same name in one function. But it is ok (rather required) to create
  // them in different functions if common block of the same name has been used
  // there.
  commonBlockMap.clear();
}

// Helper function to create a DIImportedEntityAttr for an imported declaration.
// Looks up the DIGlobalVariable for the given symbol and creates an imported
// declaration with the optional local name (for renames).
// Returns std::nullopt if the symbol's DIGlobalVariable is not found.
std::optional<mlir::LLVM::DIImportedEntityAttr>
AddDebugInfoPass::createImportedDeclForGlobal(
    llvm::StringRef symbolName, mlir::LLVM::DISubprogramAttr spAttr,
    mlir::LLVM::DIFileAttr fileAttr, mlir::StringAttr localNameAttr,
    mlir::SymbolTable *symbolTable) {
  mlir::MLIRContext *context = &getContext();
  if (auto gvAttr = lookupDIGlobalVariable(symbolName, symbolTable)) {
    return mlir::LLVM::DIImportedEntityAttr::get(
        context, llvm::dwarf::DW_TAG_imported_declaration, spAttr, *gvAttr,
        fileAttr, /*line=*/1, /*name=*/localNameAttr, /*elements*/ {});
  }
  return std::nullopt;
}

// Process USE with ONLY clause
void AddDebugInfoPass::handleOnlyClause(
    fir::UseStmtOp useOp, mlir::LLVM::DISubprogramAttr spAttr,
    mlir::LLVM::DIFileAttr fileAttr, mlir::SymbolTable *symbolTable,
    llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> &importedModules) {
  // Process ONLY symbols (without renames)
  if (auto onlySymbols = useOp.getOnlySymbols()) {
    for (mlir::Attribute attr : *onlySymbols) {
      auto symbolRef = mlir::cast<mlir::FlatSymbolRefAttr>(attr);
      if (auto importedDecl = createImportedDeclForGlobal(
              symbolRef.getValue(), spAttr, fileAttr, mlir::StringAttr(),
              symbolTable))
        importedModules.insert(*importedDecl);
    }
  }

  // Process renames within ONLY clause
  if (auto renames = useOp.getRenames()) {
    for (auto attr : *renames) {
      auto renameAttr = mlir::cast<fir::UseRenameAttr>(attr);
      if (auto importedDecl = createImportedDeclForGlobal(
              renameAttr.getSymbol().getValue(), spAttr, fileAttr,
              renameAttr.getLocalName(), symbolTable))
        importedModules.insert(*importedDecl);
    }
  }
}

// Process USE with renames but no ONLY clause
void AddDebugInfoPass::handleRenamesWithoutOnly(
    fir::UseStmtOp useOp, mlir::LLVM::DISubprogramAttr spAttr,
    mlir::LLVM::DIModuleAttr modAttr, mlir::LLVM::DIFileAttr fileAttr,
    mlir::SymbolTable *symbolTable,
    llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> &importedModules) {
  mlir::MLIRContext *context = &getContext();
  llvm::SmallVector<mlir::LLVM::DINodeAttr> childDeclarations;

  if (auto renames = useOp.getRenames()) {
    for (auto attr : *renames) {
      auto renameAttr = mlir::cast<fir::UseRenameAttr>(attr);
      if (auto importedDecl = createImportedDeclForGlobal(
              renameAttr.getSymbol().getValue(), spAttr, fileAttr,
              renameAttr.getLocalName(), symbolTable))
        childDeclarations.push_back(*importedDecl);
    }
  }

  // Create module import with renamed declarations as children
  auto moduleImport = mlir::LLVM::DIImportedEntityAttr::get(
      context, llvm::dwarf::DW_TAG_imported_module, spAttr, modAttr, fileAttr,
      /*line=*/1, /*name=*/nullptr, childDeclarations);
  importedModules.insert(moduleImport);
}

// Process all USE statements in a function and collect imported entities
void AddDebugInfoPass::handleUseStatements(
    mlir::func::FuncOp funcOp, mlir::LLVM::DISubprogramAttr spAttr,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DICompileUnitAttr cuAttr,
    mlir::SymbolTable *symbolTable,
    llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> &importedEntities) {
  mlir::MLIRContext *context = &getContext();

  funcOp.walk([&](fir::UseStmtOp useOp) {
    mlir::LLVM::DIModuleAttr modAttr = getOrCreateModuleAttr(
        useOp.getModuleName().str(), fileAttr, cuAttr, /*line=*/1,
        /*decl=*/true);

    llvm::DenseSet<mlir::LLVM::DIImportedEntityAttr> importedModules;

    if (useOp.hasOnlyClause())
      handleOnlyClause(useOp, spAttr, fileAttr, symbolTable, importedModules);
    else if (useOp.hasRenames())
      handleRenamesWithoutOnly(useOp, spAttr, modAttr, fileAttr, symbolTable,
                               importedModules);
    else {
      // Simple module import
      auto importedEntity = mlir::LLVM::DIImportedEntityAttr::get(
          context, llvm::dwarf::DW_TAG_imported_module, spAttr, modAttr,
          fileAttr, /*line=*/1, /*name=*/nullptr, /*elements*/ {});
      importedModules.insert(importedEntity);
    }

    importedEntities.insert(importedModules.begin(), importedModules.end());
  });
}

void AddDebugInfoPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::SymbolTable symbolTable(module);
  llvm::StringRef fileName;
  std::string filePath;
  std::optional<mlir::DataLayout> dl =
      fir::support::getOrSetMLIRDataLayout(module, /*allowDefaultLayout=*/true);
  if (!dl) {
    mlir::emitError(module.getLoc(), "Missing data layout attribute in module");
    signalPassFailure();
    return;
  }
  mlir::OpBuilder builder(context);
  if (dwarfVersion > 0) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    llvm::SmallVector<mlir::Attribute> moduleFlags;
    mlir::IntegerType int32Ty = mlir::IntegerType::get(context, 32);
    moduleFlags.push_back(builder.getAttr<mlir::LLVM::ModuleFlagAttr>(
        mlir::LLVM::ModFlagBehavior::Max,
        mlir::StringAttr::get(context, "Dwarf Version"),
        mlir::IntegerAttr::get(int32Ty, dwarfVersion)));
    mlir::LLVM::ModuleFlagsOp::create(builder, module.getLoc(),
                                      builder.getArrayAttr(moduleFlags));
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
      isOptimized, debugLevel,
      /*nameTableKind=*/mlir::LLVM::DINameTableKind::Default,
      splitDwarfFile.empty() ? mlir::StringAttr()
                             : mlir::StringAttr::get(context, splitDwarfFile));

  // Process module globals early.
  // Walk through all DeclareOps in functions and process globals that are
  // module variables. This ensures that when we process USE statements,
  // the DIGlobalVariable lookups will succeed.
  if (debugLevel == mlir::LLVM::DIEmissionKind::Full) {
    module.walk([&](fir::cg::XDeclareOp declOp) {
      mlir::Operation *defOp = declOp.getMemref().getDefiningOp();
      if (defOp && llvm::isa<fir::AddrOfOp>(defOp)) {
        if (auto globalOp =
                symbolTable.lookup<fir::GlobalOp>(declOp.getUniqName())) {
          // Only process module variables here, not SAVE variables
          if (isModuleVariable(globalOp)) {
            handleGlobalOp(globalOp, fileAttr, cuAttr, typeGen, &symbolTable,
                           declOp);
          }
        }
      }
    });
  }

  module.walk([&](mlir::func::FuncOp funcOp) {
    handleFuncOp(funcOp, fileAttr, cuAttr, typeGen, &symbolTable);
  });
  // We have processed all function. Attach common block variables to the
  // global that represent the storage.
  for (auto [global, exprs] : globalToGlobalExprsMap) {
    auto arrayAttr = mlir::ArrayAttr::get(context, exprs);
    global->setLoc(builder.getFusedLoc({global.getLoc()}, arrayAttr));
  }
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
