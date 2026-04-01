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
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "aiir/Transforms/RegionUtils.h"
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
                       aiir::LLVM::DIFileAttr fileAttr,
                       aiir::LLVM::DIScopeAttr scopeAttr,
                       fir::DebugTypeGenerator &typeGen,
                       aiir::SymbolTable *symbolTable, aiir::Value dummyScope);
  void handleDeclareValueOp(fir::DeclareValueOp declOp,
                            aiir::LLVM::DIFileAttr fileAttr,
                            aiir::LLVM::DIScopeAttr scopeAttr,
                            fir::DebugTypeGenerator &typeGen,
                            aiir::SymbolTable *symbolTable,
                            aiir::Value dummyScope);

public:
  AddDebugInfoPass(fir::AddDebugInfoOptions options) : Base(options) {}
  void runOnOperation() override;

private:
  llvm::StringMap<aiir::LLVM::DIModuleAttr> moduleMap;
  llvm::StringMap<aiir::LLVM::DICommonBlockAttr> commonBlockMap;
  // List of GlobalVariableExpressionAttr that are attached to a given global
  // that represents the storage for common block.
  llvm::DenseMap<fir::GlobalOp, llvm::SmallVector<aiir::Attribute>>
      globalToGlobalExprsMap;

  aiir::LLVM::DIModuleAttr getOrCreateModuleAttr(
      const std::string &name, aiir::LLVM::DIFileAttr fileAttr,
      aiir::LLVM::DIScopeAttr scope, unsigned line, bool decl);
  aiir::LLVM::DICommonBlockAttr
  getOrCreateCommonBlockAttr(llvm::StringRef name,
                             aiir::LLVM::DIFileAttr fileAttr,
                             aiir::LLVM::DIScopeAttr scope, unsigned line);

  void handleGlobalOp(fir::GlobalOp glocalOp, aiir::LLVM::DIFileAttr fileAttr,
                      aiir::LLVM::DIScopeAttr scope,
                      fir::DebugTypeGenerator &typeGen,
                      aiir::SymbolTable *symbolTable,
                      fir::cg::XDeclareOp declOp);
  void handleFuncOp(aiir::func::FuncOp funcOp, aiir::LLVM::DIFileAttr fileAttr,
                    aiir::LLVM::DICompileUnitAttr cuAttr,
                    fir::DebugTypeGenerator &typeGen,
                    aiir::SymbolTable *symbolTable);
  void handleOnlyClause(
      fir::UseStmtOp useOp, aiir::LLVM::DISubprogramAttr spAttr,
      aiir::LLVM::DIFileAttr fileAttr, aiir::SymbolTable *symbolTable,
      llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> &importedModules);
  void handleRenamesWithoutOnly(
      fir::UseStmtOp useOp, aiir::LLVM::DISubprogramAttr spAttr,
      aiir::LLVM::DIModuleAttr modAttr, aiir::LLVM::DIFileAttr fileAttr,
      aiir::SymbolTable *symbolTable,
      llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> &importedModules);
  void handleUseStatements(
      aiir::func::FuncOp funcOp, aiir::LLVM::DISubprogramAttr spAttr,
      aiir::LLVM::DIFileAttr fileAttr, aiir::LLVM::DICompileUnitAttr cuAttr,
      aiir::SymbolTable *symbolTable,
      llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> &importedEntities);
  std::optional<aiir::LLVM::DIImportedEntityAttr> createImportedDeclForGlobal(
      llvm::StringRef symbolName, aiir::LLVM::DISubprogramAttr spAttr,
      aiir::LLVM::DIFileAttr fileAttr, aiir::StringAttr localNameAttr,
      aiir::SymbolTable *symbolTable);
  bool createCommonBlockGlobal(fir::cg::XDeclareOp declOp,
                               const std::string &name,
                               aiir::LLVM::DIFileAttr fileAttr,
                               aiir::LLVM::DIScopeAttr scopeAttr,
                               fir::DebugTypeGenerator &typeGen,
                               aiir::SymbolTable *symbolTable);
  std::optional<aiir::LLVM::DIModuleAttr>
  getModuleAttrFromGlobalOp(fir::GlobalOp globalOp,
                            aiir::LLVM::DIFileAttr fileAttr,
                            aiir::LLVM::DIScopeAttr scope);

  template <typename Op>
  void handleLocalVariable(Op declOp, llvm::StringRef name,
                           aiir::LLVM::DIFileAttr fileAttr,
                           aiir::LLVM::DIScopeAttr scopeAttr,
                           fir::DebugTypeGenerator &typeGen,
                           aiir::Value dummyScope, aiir::Type typeToConvert,
                           fir::cg::XDeclareOp typeGenDeclOp);
};

bool debugInfoIsAlreadySet(aiir::Location loc) {
  if (aiir::isa<aiir::FusedLoc>(loc)) {
    if (loc->findInstanceOf<aiir::FusedLocWith<fir::LocationKindAttr>>())
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
aiir::StringAttr getTargetFunctionName(aiir::AIIRContext *context,
                                       aiir::Location Loc,
                                       llvm::StringRef parentName) {
  auto fileLoc = Loc->findInstanceOf<aiir::FileLineColLoc>();

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
  return aiir::StringAttr::get(
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
static std::optional<aiir::LLVM::DIGlobalVariableAttr>
lookupDIGlobalVariable(llvm::StringRef symbolName,
                       aiir::SymbolTable *symbolTable) {
  if (auto globalOp = symbolTable->lookup<fir::GlobalOp>(symbolName)) {
    if (auto fusedLoc = aiir::dyn_cast<aiir::FusedLoc>(globalOp.getLoc())) {
      if (auto metadata = fusedLoc.getMetadata()) {
        if (auto arrayAttr = aiir::dyn_cast<aiir::ArrayAttr>(metadata)) {
          for (auto elem : arrayAttr) {
            if (auto gvExpr =
                    aiir::dyn_cast<aiir::LLVM::DIGlobalVariableExpressionAttr>(
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
    aiir::LLVM::DIFileAttr fileAttr, aiir::LLVM::DIScopeAttr scopeAttr,
    fir::DebugTypeGenerator &typeGen, aiir::SymbolTable *symbolTable) {
  aiir::AIIRContext *context = &getContext();
  aiir::OpBuilder builder(context);

  std::optional<std::int64_t> offset;
  aiir::Value storage = declOp.getStorage();
  if (!storage)
    return false;

  // Extract offset from storage_offset attribute
  uint64_t storageOffset = declOp.getStorageOffset();
  if (storageOffset != 0)
    offset = static_cast<std::int64_t>(storageOffset);

  // Get the GlobalOp from the storage value.
  // The storage may be wrapped in ConvertOp, so unwrap it first.
  aiir::Operation *storageOp = storage.getDefiningOp();
  if (auto convertOp = aiir::dyn_cast_if_present<fir::ConvertOp>(storageOp))
    storageOp = convertOp.getValue().getDefiningOp();

  auto addrOfOp = aiir::dyn_cast_if_present<fir::AddrOfOp>(storageOp);
  if (!addrOfOp)
    return false;

  aiir::SymbolRefAttr sym = addrOfOp.getSymbol();
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
  aiir::LLVM::DICommonBlockAttr commonBlock =
      getOrCreateCommonBlockAttr(commonName, fileAttr, scopeAttr, line);

  aiir::LLVM::DITypeAttr diType = typeGen.convertType(
      fir::unwrapRefType(declOp.getType()), fileAttr, scopeAttr, declOp);

  line = getLineFromLoc(declOp.getLoc());
  auto gvAttr = aiir::LLVM::DIGlobalVariableAttr::get(
      context, commonBlock, aiir::StringAttr::get(context, name),
      declOp.getUniqName(), fileAttr, line, diType,
      /*isLocalToUnit*/ false, /*isDefinition*/ true, /* alignInBits*/ 0);

  // Create DIExpression for offset if needed
  aiir::LLVM::DIExpressionAttr expr;
  if (offset && *offset != 0) {
    llvm::SmallVector<aiir::LLVM::DIExpressionElemAttr> ops;
    ops.push_back(aiir::LLVM::DIExpressionElemAttr::get(
        context, llvm::dwarf::DW_OP_plus_uconst, *offset));
    expr = aiir::LLVM::DIExpressionAttr::get(context, ops);
  }

  auto dbgExpr = aiir::LLVM::DIGlobalVariableExpressionAttr::get(
      global.getContext(), gvAttr, expr);
  globalToGlobalExprsMap[global].push_back(dbgExpr);

  return true;
}

template <typename Op>
void AddDebugInfoPass::handleLocalVariable(Op declOp, llvm::StringRef name,
                                           aiir::LLVM::DIFileAttr fileAttr,
                                           aiir::LLVM::DIScopeAttr scopeAttr,
                                           fir::DebugTypeGenerator &typeGen,
                                           aiir::Value dummyScope,
                                           aiir::Type typeToConvert,
                                           fir::cg::XDeclareOp typeGenDeclOp) {
  aiir::AIIRContext *context = &getContext();
  aiir::OpBuilder builder(context);

  // Get the dummy argument position from the explicit attribute.
  unsigned argNo = 0;
  if (dummyScope && declOp.getDummyScope() == dummyScope) {
    if (auto argNoOpt = declOp.getDummyArgNo())
      argNo = *argNoOpt;
  }

  auto tyAttr =
      typeGen.convertType(typeToConvert, fileAttr, scopeAttr, typeGenDeclOp);

  auto localVarAttr = aiir::LLVM::DILocalVariableAttr::get(
      context, scopeAttr, aiir::StringAttr::get(context, name), fileAttr,
      getLineFromLoc(declOp.getLoc()), argNo, /* alignInBits*/ 0, tyAttr,
      aiir::LLVM::DIFlags::Zero);
  declOp->setLoc(builder.getFusedLoc({declOp->getLoc()}, localVarAttr));
}

void AddDebugInfoPass::handleDeclareOp(fir::cg::XDeclareOp declOp,
                                       aiir::LLVM::DIFileAttr fileAttr,
                                       aiir::LLVM::DIScopeAttr scopeAttr,
                                       fir::DebugTypeGenerator &typeGen,
                                       aiir::SymbolTable *symbolTable,
                                       aiir::Value dummyScope) {
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  if (createCommonBlockGlobal(declOp, result.second.name, fileAttr, scopeAttr,
                              typeGen, symbolTable))
    return;

  // If this DeclareOp actually represents a global then treat it as such.
  aiir::Operation *defOp = declOp.getMemref().getDefiningOp();
  if (defOp && llvm::isa<fir::AddrOfOp>(defOp)) {
    if (auto global =
            symbolTable->lookup<fir::GlobalOp>(declOp.getUniqName())) {
      handleGlobalOp(global, fileAttr, scopeAttr, typeGen, symbolTable, declOp);
      return;
    }
  }

  handleLocalVariable(declOp, result.second.name, fileAttr, scopeAttr, typeGen,
                      dummyScope, fir::unwrapRefType(declOp.getType()), declOp);
}

void AddDebugInfoPass::handleDeclareValueOp(fir::DeclareValueOp declOp,
                                            aiir::LLVM::DIFileAttr fileAttr,
                                            aiir::LLVM::DIScopeAttr scopeAttr,
                                            fir::DebugTypeGenerator &typeGen,
                                            aiir::SymbolTable *symbolTable,
                                            aiir::Value dummyScope) {
  auto result = fir::NameUniquer::deconstruct(declOp.getUniqName());

  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  handleLocalVariable(declOp, result.second.name, fileAttr, scopeAttr, typeGen,
                      dummyScope, declOp.getValue().getType(), nullptr);
}

aiir::LLVM::DICommonBlockAttr AddDebugInfoPass::getOrCreateCommonBlockAttr(
    llvm::StringRef name, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, unsigned line) {
  aiir::AIIRContext *context = &getContext();
  aiir::LLVM::DICommonBlockAttr cbAttr;
  if (auto iter{commonBlockMap.find(name)}; iter != commonBlockMap.end()) {
    cbAttr = iter->getValue();
  } else {
    cbAttr = aiir::LLVM::DICommonBlockAttr::get(
        context, scope, nullptr, aiir::StringAttr::get(context, name), fileAttr,
        line);
    commonBlockMap[name] = cbAttr;
  }
  return cbAttr;
}

// The `module` does not have a first class representation in the `FIR`. We
// extract information about it from the name of the identifiers and keep a
// map to avoid duplication.
aiir::LLVM::DIModuleAttr AddDebugInfoPass::getOrCreateModuleAttr(
    const std::string &name, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, unsigned line, bool decl) {
  aiir::AIIRContext *context = &getContext();
  aiir::LLVM::DIModuleAttr modAttr;
  if (auto iter{moduleMap.find(name)}; iter != moduleMap.end()) {
    modAttr = iter->getValue();
  } else {
    // When decl is true, it means that module is only being used in this
    // compilation unit and it is defined elsewhere. But if the file/line/scope
    // fields are valid, the module is not merged with its definition and is
    // considered different. So we only set those fields when decl is false.
    modAttr = aiir::LLVM::DIModuleAttr::get(
        context, decl ? nullptr : fileAttr, decl ? nullptr : scope,
        aiir::StringAttr::get(context, name),
        /* configMacros */ aiir::StringAttr(),
        /* includePath */ aiir::StringAttr(),
        /* apinotes */ aiir::StringAttr(), decl ? 0 : line, decl);
    moduleMap[name] = modAttr;
  }
  return modAttr;
}

/// If globalOp represents a module variable, return a ModuleAttr that
/// represents that module.
std::optional<aiir::LLVM::DIModuleAttr>
AddDebugInfoPass::getModuleAttrFromGlobalOp(fir::GlobalOp globalOp,
                                            aiir::LLVM::DIFileAttr fileAttr,
                                            aiir::LLVM::DIScopeAttr scope) {
  aiir::AIIRContext *context = &getContext();
  aiir::OpBuilder builder(context);

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

  aiir::LLVM::DISubprogramAttr sp =
      aiir::dyn_cast_if_present<aiir::LLVM::DISubprogramAttr>(scope);
  // Modules are generated at compile unit scope
  if (sp)
    scope = sp.getCompileUnit();

  return getOrCreateModuleAttr(result.second.modules[0], fileAttr, scope,
                               std::max(line - 1, (unsigned)1),
                               !globalOp.isInitialized());
}

void AddDebugInfoPass::handleGlobalOp(fir::GlobalOp globalOp,
                                      aiir::LLVM::DIFileAttr fileAttr,
                                      aiir::LLVM::DIScopeAttr scope,
                                      fir::DebugTypeGenerator &typeGen,
                                      aiir::SymbolTable *symbolTable,
                                      fir::cg::XDeclareOp declOp) {
  if (debugInfoIsAlreadySet(globalOp.getLoc()))
    return;
  aiir::AIIRContext *context = &getContext();
  aiir::OpBuilder builder(context);

  std::pair result = fir::NameUniquer::deconstruct(globalOp.getSymName());
  if (result.first != fir::NameUniquer::NameKind::VARIABLE)
    return;

  if (fir::NameUniquer::isSpecialSymbol(result.second.name))
    return;

  unsigned line = getLineFromLoc(globalOp.getLoc());
  std::optional<aiir::LLVM::DIModuleAttr> modOpt =
      getModuleAttrFromGlobalOp(globalOp, fileAttr, scope);
  if (modOpt)
    scope = *modOpt;

  aiir::LLVM::DITypeAttr diType =
      typeGen.convertType(globalOp.getType(), fileAttr, scope, declOp);
  auto gvAttr = aiir::LLVM::DIGlobalVariableAttr::get(
      context, scope, aiir::StringAttr::get(context, result.second.name),
      aiir::StringAttr::get(context, globalOp.getName()), fileAttr, line,
      diType, /*isLocalToUnit*/ false,
      /*isDefinition*/ globalOp.isInitialized(), /* alignInBits*/ 0);
  auto dbgExpr = aiir::LLVM::DIGlobalVariableExpressionAttr::get(
      globalOp.getContext(), gvAttr, nullptr);
  auto arrayAttr = aiir::ArrayAttr::get(context, {dbgExpr});
  globalOp->setLoc(builder.getFusedLoc({globalOp.getLoc()}, arrayAttr));
}

static aiir::LLVM::DISubprogramAttr
getScope(aiir::Operation *op, aiir::LLVM::DISubprogramAttr defaultScope) {
  if (auto tOp = op->getParentOfType<aiir::omp::TargetOp>()) {
    if (auto fusedLoc = llvm::dyn_cast<aiir::FusedLoc>(tOp.getLoc())) {
      if (auto sp = llvm::dyn_cast<aiir::LLVM::DISubprogramAttr>(
              fusedLoc.getMetadata()))
        return sp;
    }
  }
  return defaultScope;
}

void AddDebugInfoPass::handleFuncOp(aiir::func::FuncOp funcOp,
                                    aiir::LLVM::DIFileAttr fileAttr,
                                    aiir::LLVM::DICompileUnitAttr cuAttr,
                                    fir::DebugTypeGenerator &typeGen,
                                    aiir::SymbolTable *symbolTable) {
  aiir::Location l = funcOp->getLoc();
  // If fused location has already been created then nothing to do
  // Otherwise, create a fused location.
  if (debugInfoIsAlreadySet(l))
    return;

  aiir::AIIRContext *context = &getContext();
  aiir::OpBuilder builder(context);
  llvm::StringRef fileName(fileAttr.getName());
  llvm::StringRef filePath(fileAttr.getDirectory());
  unsigned int CC = (funcOp.getName() == fir::NameUniquer::doProgramEntry())
                        ? llvm::dwarf::getCallingConvention("DW_CC_program")
                        : llvm::dwarf::getCallingConvention("DW_CC_normal");

  if (auto funcLoc = aiir::dyn_cast<aiir::FileLineColLoc>(l)) {
    fileName = llvm::sys::path::filename(funcLoc.getFilename().getValue());
    filePath = llvm::sys::path::parent_path(funcLoc.getFilename().getValue());
  }

  aiir::StringAttr fullName = aiir::StringAttr::get(context, funcOp.getName());
  aiir::Attribute attr = funcOp->getAttr(fir::getInternalFuncNameAttrName());
  aiir::StringAttr funcName =
      (attr) ? aiir::cast<aiir::StringAttr>(attr)
             : aiir::StringAttr::get(context, funcOp.getName());

  auto result = fir::NameUniquer::deconstruct(funcName);
  funcName = aiir::StringAttr::get(context, result.second.name);

  // try to use a better function name than _QQmain for the program statement
  bool isMain = false;
  if (funcName == fir::NameUniquer::doProgramEntry()) {
    isMain = true;
    aiir::StringAttr bindcName =
        funcOp->getAttrOfType<aiir::StringAttr>(fir::getSymbolAttrName());
    if (bindcName)
      funcName = bindcName;
  }

  llvm::SmallVector<aiir::LLVM::DITypeAttr> types;
  for (auto resTy : funcOp.getResultTypes()) {
    auto tyAttr =
        typeGen.convertType(resTy, fileAttr, cuAttr, /*declOp=*/nullptr);
    types.push_back(tyAttr);
  }
  // If no return type then add a null type as a place holder for that.
  if (types.empty())
    types.push_back(aiir::LLVM::DINullTypeAttr::get(context));
  for (auto inTy : funcOp.getArgumentTypes()) {
    auto tyAttr = typeGen.convertType(fir::unwrapRefType(inTy), fileAttr,
                                      cuAttr, /*declOp=*/nullptr);
    types.push_back(tyAttr);
  }

  aiir::LLVM::DISubroutineTypeAttr subTypeAttr =
      aiir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
  aiir::LLVM::DIFileAttr funcFileAttr =
      aiir::LLVM::DIFileAttr::get(context, fileName, filePath);

  // Only definitions need a distinct identifier and a compilation unit.
  aiir::DistinctAttr id, id2;
  aiir::LLVM::DIScopeAttr Scope = fileAttr;
  aiir::LLVM::DICompileUnitAttr compilationUnit;
  aiir::LLVM::DISubprogramFlags subprogramFlags =
      aiir::LLVM::DISubprogramFlags{};
  if (isOptimized)
    subprogramFlags = aiir::LLVM::DISubprogramFlags::Optimized;
  if (isMain)
    subprogramFlags =
        subprogramFlags | aiir::LLVM::DISubprogramFlags::MainSubprogram;
  if (!funcOp.isExternal()) {
    // Place holder and final function have to have different IDs, otherwise
    // translation code will reject one of them.
    id = aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
    id2 = aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
    compilationUnit = cuAttr;
    subprogramFlags =
        subprogramFlags | aiir::LLVM::DISubprogramFlags::Definition;
  }

  // Check if the function has the pure, elemental, or recursive procedure
  // attribute
  if (fir::hasProcedureAttr<fir::FortranProcedureFlagsEnum::pure>(funcOp))
    subprogramFlags = subprogramFlags | aiir::LLVM::DISubprogramFlags::Pure;

  if (fir::hasProcedureAttr<fir::FortranProcedureFlagsEnum::elemental>(funcOp))
    subprogramFlags =
        subprogramFlags | aiir::LLVM::DISubprogramFlags::Elemental;

  if (fir::hasProcedureAttr<fir::FortranProcedureFlagsEnum::recursive>(funcOp))
    subprogramFlags =
        subprogramFlags | aiir::LLVM::DISubprogramFlags::Recursive;

  unsigned line = getLineFromLoc(l);
  if (fir::isInternalProcedure(funcOp)) {
    // For contained functions, the scope is the parent subroutine.
    aiir::SymbolRefAttr sym = aiir::cast<aiir::SymbolRefAttr>(
        funcOp->getAttr(fir::getHostSymbolAttrName()));
    if (sym) {
      if (auto func =
              symbolTable->lookup<aiir::func::FuncOp>(sym.getLeafReference())) {
        // Make sure that parent is processed.
        handleFuncOp(func, fileAttr, cuAttr, typeGen, symbolTable);
        if (auto fusedLoc =
                aiir::dyn_cast_if_present<aiir::FusedLoc>(func.getLoc())) {
          if (auto spAttr =
                  aiir::dyn_cast_if_present<aiir::LLVM::DISubprogramAttr>(
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
                             llvm::ArrayRef<aiir::LLVM::DINodeAttr> entities) {
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
    funcOp.walk([&](aiir::omp::TargetOp targetOp) {
      unsigned line = getLineFromLoc(targetOp.getLoc());
      aiir::StringAttr name =
          getTargetFunctionName(context, targetOp.getLoc(), funcOp.getName());
      aiir::LLVM::DISubprogramFlags flags =
          aiir::LLVM::DISubprogramFlags::Definition |
          aiir::LLVM::DISubprogramFlags::LocalToUnit;
      if (isOptimized)
        flags = flags | aiir::LLVM::DISubprogramFlags::Optimized;

      aiir::DistinctAttr id =
          aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
      llvm::SmallVector<aiir::LLVM::DITypeAttr> types;
      types.push_back(aiir::LLVM::DINullTypeAttr::get(context));
      for (auto arg : targetOp.getRegion().getArguments()) {
        auto tyAttr = typeGen.convertType(fir::unwrapRefType(arg.getType()),
                                          fileAttr, cuAttr, /*declOp=*/nullptr);
        types.push_back(tyAttr);
      }
      CC = llvm::dwarf::getCallingConvention("DW_CC_normal");
      aiir::LLVM::DISubroutineTypeAttr spTy =
          aiir::LLVM::DISubroutineTypeAttr::get(context, CC, types);
      if (lineTableOnly || entities.empty()) {
        auto spAttr = aiir::LLVM::DISubprogramAttr::get(
            context, id, compilationUnit, Scope, name, name, funcFileAttr, line,
            line, flags, spTy, /*retainedNodes=*/{}, /*annotations=*/{});
        targetOp->setLoc(builder.getFusedLoc({targetOp.getLoc()}, spAttr));
        return;
      }
      aiir::DistinctAttr recId =
          aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
      auto spAttr = aiir::LLVM::DISubprogramAttr::get(
          context, recId, /*isRecSelf=*/true, id, compilationUnit, Scope, name,
          name, funcFileAttr, line, line, flags, spTy, /*retainedNodes=*/{},
          /*annotations=*/{});

      // Make sure that information about the imported modules is copied in the
      // new function.
      llvm::SmallVector<aiir::LLVM::DINodeAttr> opEntities;
      for (aiir::LLVM::DINodeAttr N : entities) {
        if (auto entity = aiir::dyn_cast<aiir::LLVM::DIImportedEntityAttr>(N)) {
          auto importedEntity = aiir::LLVM::DIImportedEntityAttr::get(
              context, entity.getTag(), spAttr, entity.getEntity(),
              entity.getFile(), entity.getLine(), entity.getName(),
              entity.getElements());
          opEntities.push_back(importedEntity);
        }
      }

      id = aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
      spAttr = aiir::LLVM::DISubprogramAttr::get(
          context, recId, /*isRecSelf=*/false, id, compilationUnit, Scope, name,
          name, funcFileAttr, line, line, flags, spTy, opEntities,
          /*annotations=*/{});
      targetOp->setLoc(builder.getFusedLoc({targetOp.getLoc()}, spAttr));
    });
  };

  // Don't process variables if user asked for line tables only.
  if (debugLevel == aiir::LLVM::DIEmissionKind::LineTablesOnly) {
    auto spAttr = aiir::LLVM::DISubprogramAttr::get(
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
    return aiir::WalkResult::interrupt();
  });

  aiir::LLVM::DISubprogramAttr spAttr;
  llvm::SmallVector<aiir::LLVM::DINodeAttr> retainedNodes;

  if (hasUseStmts) {
    aiir::DistinctAttr recId =
        aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
    // The debug attribute in AIIR are readonly once created. But in case of
    // imported entities, we have a circular dependency. The
    // DIImportedEntityAttr requires scope information (DISubprogramAttr in this
    // case) and DISubprogramAttr requires the list of imported entities. The
    // AIIR provides a way where a DISubprogramAttr an be created with a certain
    // recID and be used in places like DIImportedEntityAttr. After that another
    // DISubprogramAttr can be created with same recID but with list of entities
    // now available. The AIIR translation code takes care of updating the
    // references. Note that references will be updated only in the things that
    // are part of DISubprogramAttr (like DIImportedEntityAttr) so we have to
    // create the final DISubprogramAttr before we process local variables.
    // Look at DIRecursiveTypeAttrInterface for more details.
    spAttr = aiir::LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/true, id, compilationUnit, Scope,
        funcName, fullName, funcFileAttr, line, line, subprogramFlags,
        subTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});

    // Process USE statements (module globals are already processed)
    llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> importedEntities;
    handleUseStatements(funcOp, spAttr, fileAttr, cuAttr, symbolTable,
                        importedEntities);

    retainedNodes.append(importedEntities.begin(), importedEntities.end());

    // Create final DISubprogramAttr with imported entities and same recId
    spAttr = aiir::LLVM::DISubprogramAttr::get(
        context, recId, /*isRecSelf=*/false, id2, compilationUnit, Scope,
        funcName, fullName, funcFileAttr, line, line, subprogramFlags,
        subTypeAttr, retainedNodes, /*annotations=*/{});
  } else
    // No USE statements - create final DISubprogramAttr directly
    spAttr = aiir::LLVM::DISubprogramAttr::get(
        context, id, compilationUnit, Scope, funcName, fullName, funcFileAttr,
        line, line, subprogramFlags, subTypeAttr, /*retainedNodes=*/{},
        /*annotations=*/{});

  funcOp->setLoc(builder.getFusedLoc({l}, spAttr));
  addTargetOpDISP(/*lineTableOnly=*/false, retainedNodes);

  // Find the first dummy_scope definition. This is the one of the current
  // function. The other ones may come from inlined calls. The variables inside
  // those inlined calls should not be identified as arguments of the current
  // function.
  aiir::Value dummyScope;
  funcOp.walk([&](fir::UndefOp undef) -> aiir::WalkResult {
    // TODO: delay fir.dummy_scope translation to undefined until
    // codegeneration. This is nicer and safer to match.
    if (llvm::isa<fir::DummyScopeType>(undef.getType())) {
      dummyScope = undef;
      return aiir::WalkResult::interrupt();
    }
    return aiir::WalkResult::advance();
  });

  funcOp.walk([&](fir::cg::XDeclareOp declOp) {
    aiir::LLVM::DISubprogramAttr spTy = getScope(declOp, spAttr);
    handleDeclareOp(declOp, fileAttr, spTy, typeGen, symbolTable, dummyScope);
  });
  funcOp.walk([&](fir::DeclareValueOp declOp) {
    aiir::LLVM::DISubprogramAttr spTy = getScope(declOp, spAttr);
    handleDeclareValueOp(declOp, fileAttr, spTy, typeGen, symbolTable,
                         dummyScope);
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
std::optional<aiir::LLVM::DIImportedEntityAttr>
AddDebugInfoPass::createImportedDeclForGlobal(
    llvm::StringRef symbolName, aiir::LLVM::DISubprogramAttr spAttr,
    aiir::LLVM::DIFileAttr fileAttr, aiir::StringAttr localNameAttr,
    aiir::SymbolTable *symbolTable) {
  aiir::AIIRContext *context = &getContext();
  if (auto gvAttr = lookupDIGlobalVariable(symbolName, symbolTable)) {
    return aiir::LLVM::DIImportedEntityAttr::get(
        context, llvm::dwarf::DW_TAG_imported_declaration, spAttr, *gvAttr,
        fileAttr, /*line=*/1, /*name=*/localNameAttr, /*elements*/ {});
  }
  return std::nullopt;
}

// Process USE with ONLY clause
void AddDebugInfoPass::handleOnlyClause(
    fir::UseStmtOp useOp, aiir::LLVM::DISubprogramAttr spAttr,
    aiir::LLVM::DIFileAttr fileAttr, aiir::SymbolTable *symbolTable,
    llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> &importedModules) {
  // Process ONLY symbols (without renames)
  if (auto onlySymbols = useOp.getOnlySymbols()) {
    for (aiir::Attribute attr : *onlySymbols) {
      auto symbolRef = aiir::cast<aiir::FlatSymbolRefAttr>(attr);
      if (auto importedDecl = createImportedDeclForGlobal(
              symbolRef.getValue(), spAttr, fileAttr, aiir::StringAttr(),
              symbolTable))
        importedModules.insert(*importedDecl);
    }
  }

  // Process renames within ONLY clause
  if (auto renames = useOp.getRenames()) {
    for (auto attr : *renames) {
      auto renameAttr = aiir::cast<fir::UseRenameAttr>(attr);
      if (auto importedDecl = createImportedDeclForGlobal(
              renameAttr.getSymbol().getValue(), spAttr, fileAttr,
              renameAttr.getLocalName(), symbolTable))
        importedModules.insert(*importedDecl);
    }
  }
}

// Process USE with renames but no ONLY clause
void AddDebugInfoPass::handleRenamesWithoutOnly(
    fir::UseStmtOp useOp, aiir::LLVM::DISubprogramAttr spAttr,
    aiir::LLVM::DIModuleAttr modAttr, aiir::LLVM::DIFileAttr fileAttr,
    aiir::SymbolTable *symbolTable,
    llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> &importedModules) {
  aiir::AIIRContext *context = &getContext();
  llvm::SmallVector<aiir::LLVM::DINodeAttr> childDeclarations;

  if (auto renames = useOp.getRenames()) {
    for (auto attr : *renames) {
      auto renameAttr = aiir::cast<fir::UseRenameAttr>(attr);
      if (auto importedDecl = createImportedDeclForGlobal(
              renameAttr.getSymbol().getValue(), spAttr, fileAttr,
              renameAttr.getLocalName(), symbolTable))
        childDeclarations.push_back(*importedDecl);
    }
  }

  // Create module import with renamed declarations as children
  auto moduleImport = aiir::LLVM::DIImportedEntityAttr::get(
      context, llvm::dwarf::DW_TAG_imported_module, spAttr, modAttr, fileAttr,
      /*line=*/1, /*name=*/nullptr, childDeclarations);
  importedModules.insert(moduleImport);
}

// Process all USE statements in a function and collect imported entities
void AddDebugInfoPass::handleUseStatements(
    aiir::func::FuncOp funcOp, aiir::LLVM::DISubprogramAttr spAttr,
    aiir::LLVM::DIFileAttr fileAttr, aiir::LLVM::DICompileUnitAttr cuAttr,
    aiir::SymbolTable *symbolTable,
    llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> &importedEntities) {
  aiir::AIIRContext *context = &getContext();

  funcOp.walk([&](fir::UseStmtOp useOp) {
    aiir::LLVM::DIModuleAttr modAttr = getOrCreateModuleAttr(
        useOp.getModuleName().str(), fileAttr, cuAttr, /*line=*/1,
        /*decl=*/true);

    llvm::DenseSet<aiir::LLVM::DIImportedEntityAttr> importedModules;

    if (useOp.hasOnlyClause())
      handleOnlyClause(useOp, spAttr, fileAttr, symbolTable, importedModules);
    else if (useOp.hasRenames())
      handleRenamesWithoutOnly(useOp, spAttr, modAttr, fileAttr, symbolTable,
                               importedModules);
    else {
      // Simple module import
      auto importedEntity = aiir::LLVM::DIImportedEntityAttr::get(
          context, llvm::dwarf::DW_TAG_imported_module, spAttr, modAttr,
          fileAttr, /*line=*/1, /*name=*/nullptr, /*elements*/ {});
      importedModules.insert(importedEntity);
    }

    importedEntities.insert(importedModules.begin(), importedModules.end());
  });
}

void AddDebugInfoPass::runOnOperation() {
  aiir::ModuleOp module = getOperation();
  aiir::AIIRContext *context = &getContext();
  aiir::SymbolTable symbolTable(module);
  llvm::StringRef fileName;
  std::string filePath;
  std::optional<aiir::DataLayout> dl =
      fir::support::getOrSetAIIRDataLayout(module, /*allowDefaultLayout=*/true);
  if (!dl) {
    aiir::emitError(module.getLoc(), "Missing data layout attribute in module");
    signalPassFailure();
    return;
  }
  aiir::OpBuilder builder(context);
  if (dwarfVersion > 0) {
    aiir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    llvm::SmallVector<aiir::Attribute> moduleFlags;
    aiir::IntegerType int32Ty = aiir::IntegerType::get(context, 32);
    moduleFlags.push_back(builder.getAttr<aiir::LLVM::ModuleFlagAttr>(
        aiir::LLVM::ModFlagBehavior::Max,
        aiir::StringAttr::get(context, "Dwarf Version"),
        aiir::IntegerAttr::get(int32Ty, dwarfVersion)));
    aiir::LLVM::ModuleFlagsOp::create(builder, module.getLoc(),
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
    if (auto fileLoc = aiir::dyn_cast<aiir::FileLineColLoc>(module.getLoc())) {
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

  aiir::LLVM::DIFileAttr fileAttr =
      aiir::LLVM::DIFileAttr::get(context, fileName, filePath);
  // Match Clang style by starting with the full compiler version and
  // appending -dwarf-debug-flags content when provided.
  std::string producerString = Fortran::common::getFlangFullVersion();
  if (!dwarfDebugFlags.empty())
    producerString += " " + dwarfDebugFlags;
  aiir::StringAttr producer = aiir::StringAttr::get(context, producerString);
  aiir::LLVM::DICompileUnitAttr cuAttr = aiir::LLVM::DICompileUnitAttr::get(
      aiir::DistinctAttr::create(aiir::UnitAttr::get(context)),
      llvm::dwarf::getLanguage("DW_LANG_Fortran95"), fileAttr, producer,
      isOptimized, debugLevel, debugInfoForProfiling,
      /*nameTableKind=*/aiir::LLVM::DINameTableKind::Default,
      splitDwarfFile.empty() ? aiir::StringAttr()
                             : aiir::StringAttr::get(context, splitDwarfFile));

  // Process module globals early.
  // Walk through all DeclareOps in functions and process globals that are
  // module variables. This ensures that when we process USE statements,
  // the DIGlobalVariable lookups will succeed.
  if (debugLevel == aiir::LLVM::DIEmissionKind::Full) {
    module.walk([&](fir::cg::XDeclareOp declOp) {
      aiir::Operation *defOp = declOp.getMemref().getDefiningOp();
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

  module.walk([&](aiir::func::FuncOp funcOp) {
    handleFuncOp(funcOp, fileAttr, cuAttr, typeGen, &symbolTable);
  });
  // We have processed all function. Attach common block variables to the
  // global that represent the storage.
  for (auto [global, exprs] : globalToGlobalExprsMap) {
    auto arrayAttr = aiir::ArrayAttr::get(context, exprs);
    global->setLoc(builder.getFusedLoc({global.getLoc()}, arrayAttr));
  }
  // Process any global which was not processed through DeclareOp.
  if (debugLevel == aiir::LLVM::DIEmissionKind::Full) {
    // Process 'GlobalOp' only if full debug info is requested.
    for (auto globalOp : module.getOps<fir::GlobalOp>())
      handleGlobalOp(globalOp, fileAttr, cuAttr, typeGen, &symbolTable,
                     /*declOp=*/nullptr);
  }
}

std::unique_ptr<aiir::Pass>
fir::createAddDebugInfoPass(fir::AddDebugInfoOptions options) {
  return std::make_unique<AddDebugInfoPass>(options);
}
