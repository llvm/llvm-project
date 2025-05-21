//===--- CIRGenModule.h - Per-Module state for CIR gen ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H

#include "CIRGenBuilder.h"
#include "CIRGenTypeCache.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"

#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "TargetInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
class ASTContext;
class CodeGenOptions;
class Decl;
class GlobalDecl;
class LangOptions;
class TargetInfo;
class VarDecl;

namespace CIRGen {

class CIRGenFunction;
class CIRGenCXXABI;

enum ForDefinition_t : bool { NotForDefinition = false, ForDefinition = true };

/// This class organizes the cross-function state that is used while generating
/// CIR code.
class CIRGenModule : public CIRGenTypeCache {
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;

public:
  CIRGenModule(mlir::MLIRContext &mlirContext, clang::ASTContext &astContext,
               const clang::CodeGenOptions &cgo,
               clang::DiagnosticsEngine &diags);

  ~CIRGenModule();

private:
  mutable std::unique_ptr<TargetCIRGenInfo> theTargetCIRGenInfo;

  CIRGenBuilderTy builder;

  /// Hold Clang AST information.
  clang::ASTContext &astContext;

  const clang::LangOptions &langOpts;

  const clang::CodeGenOptions &codeGenOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  clang::DiagnosticsEngine &diags;

  const clang::TargetInfo &target;

  std::unique_ptr<CIRGenCXXABI> abi;

  CIRGenTypes genTypes;

  /// Per-function codegen information. Updated everytime emitCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *curCGF = nullptr;

public:
  mlir::ModuleOp getModule() const { return theModule; }
  CIRGenBuilderTy &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() const { return astContext; }
  const clang::TargetInfo &getTarget() const { return target; }
  const clang::CodeGenOptions &getCodeGenOpts() const { return codeGenOpts; }
  CIRGenTypes &getTypes() { return genTypes; }
  const clang::LangOptions &getLangOpts() const { return langOpts; }

  CIRGenCXXABI &getCXXABI() const { return *abi; }
  mlir::MLIRContext &getMLIRContext() { return *builder.getContext(); }

  const cir::CIRDataLayout getDataLayout() const {
    // FIXME(cir): instead of creating a CIRDataLayout every time, set it as an
    // attribute for the CIRModule class.
    return cir::CIRDataLayout(theModule);
  }

  /// -------
  /// Handling globals
  /// -------

  mlir::Operation *getGlobalValue(llvm::StringRef ref);

  /// If the specified mangled name is not in the module, create and return an
  /// mlir::GlobalOp value
  cir::GlobalOp getOrCreateCIRGlobal(llvm::StringRef mangledName, mlir::Type ty,
                                     LangAS langAS, const VarDecl *d,
                                     ForDefinition_t isForDefinition);

  cir::GlobalOp getOrCreateCIRGlobal(const VarDecl *d, mlir::Type ty,
                                     ForDefinition_t isForDefinition);

  static cir::GlobalOp createGlobalOp(CIRGenModule &cgm, mlir::Location loc,
                                      llvm::StringRef name, mlir::Type t,
                                      mlir::Operation *insertPoint = nullptr);

  llvm::StringMap<unsigned> cgGlobalNames;
  std::string getUniqueGlobalName(const std::string &baseName);

  /// Return the mlir::Value for the address of the given global variable.
  /// If Ty is non-null and if the global doesn't exist, then it will be created
  /// with the specified type instead of whatever the normal requested type
  /// would be. If IsForDefinition is true, it is guaranteed that an actual
  /// global with type Ty will be returned, not conversion of a variable with
  /// the same mangled name but some other type.
  mlir::Value
  getAddrOfGlobalVar(const VarDecl *d, mlir::Type ty = {},
                     ForDefinition_t isForDefinition = NotForDefinition);

  /// Return a constant array for the given string.
  mlir::Attribute getConstantArrayFromStringLiteral(const StringLiteral *e);

  /// Return a global symbol reference to a constant array for the given string
  /// literal.
  cir::GlobalOp getGlobalForStringLiteral(const StringLiteral *s,
                                          llvm::StringRef name = ".str");

  const TargetCIRGenInfo &getTargetCIRGenInfo();

  /// Helpers to convert the presumed location of Clang's SourceLocation to an
  /// MLIR Location.
  mlir::Location getLoc(clang::SourceLocation cLoc);
  mlir::Location getLoc(clang::SourceRange cRange);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  clang::CharUnits getNaturalTypeAlignment(clang::QualType t,
                                           LValueBaseInfo *baseInfo);

  void emitTopLevelDecl(clang::Decl *decl);

  bool verifyModule() const;

  /// Return the address of the given function. If funcType is non-null, then
  /// this function will use the specified type if it has to create it.
  // TODO: this is a bit weird as `GetAddr` given we give back a FuncOp?
  cir::FuncOp
  getAddrOfFunction(clang::GlobalDecl gd, mlir::Type funcType = nullptr,
                    bool forVTable = false, bool dontDefer = false,
                    ForDefinition_t isForDefinition = NotForDefinition);

  /// Emit code for a single global function or variable declaration. Forward
  /// declarations are emitted lazily.
  void emitGlobal(clang::GlobalDecl gd);

  mlir::Type convertType(clang::QualType type);

  void emitGlobalDefinition(clang::GlobalDecl gd,
                            mlir::Operation *op = nullptr);
  void emitGlobalFunctionDefinition(clang::GlobalDecl gd, mlir::Operation *op);
  void emitGlobalVarDefinition(const clang::VarDecl *vd,
                               bool isTentative = false);

  void emitGlobalOpenACCDecl(const clang::OpenACCConstructDecl *cd);

  // C++ related functions.
  void emitDeclContext(const DeclContext *dc);

  /// Return the result of value-initializing the given type, i.e. a null
  /// expression of the given type.
  mlir::Value emitNullConstant(QualType t, mlir::Location loc);

  llvm::StringRef getMangledName(clang::GlobalDecl gd);

  static void setInitializer(cir::GlobalOp &op, mlir::Attribute value);

  cir::FuncOp
  getOrCreateCIRFunction(llvm::StringRef mangledName, mlir::Type funcType,
                         clang::GlobalDecl gd, bool forVTable,
                         bool dontDefer = false, bool isThunk = false,
                         ForDefinition_t isForDefinition = NotForDefinition,
                         mlir::ArrayAttr extraAttrs = {});

  cir::FuncOp createCIRFunction(mlir::Location loc, llvm::StringRef name,
                                cir::FuncType funcType,
                                const clang::FunctionDecl *funcDecl);

  mlir::IntegerAttr getSize(CharUnits size) {
    return builder.getSizeFromCharUnits(&getMLIRContext(), size);
  }

  const llvm::Triple &getTriple() const { return target.getTriple(); }

  cir::GlobalLinkageKind getCIRLinkageForDeclarator(const DeclaratorDecl *dd,
                                                    GVALinkage linkage,
                                                    bool isConstantVariable);

  cir::GlobalLinkageKind getCIRLinkageVarDefinition(const VarDecl *vd,
                                                    bool isConstant);

  /// Helpers to emit "not yet implemented" error diagnostics
  DiagnosticBuilder errorNYI(SourceLocation, llvm::StringRef);

  template <typename T>
  DiagnosticBuilder errorNYI(SourceLocation loc, llvm::StringRef feature,
                             const T &name) {
    unsigned diagID =
        diags.getCustomDiagID(DiagnosticsEngine::Error,
                              "ClangIR code gen Not Yet Implemented: %0: %1");
    return diags.Report(loc, diagID) << feature << name;
  }

  DiagnosticBuilder errorNYI(mlir::Location loc, llvm::StringRef feature) {
    // TODO: Convert the location to a SourceLocation
    unsigned diagID = diags.getCustomDiagID(
        DiagnosticsEngine::Error, "ClangIR code gen Not Yet Implemented: %0");
    return diags.Report(diagID) << feature;
  }

  DiagnosticBuilder errorNYI(llvm::StringRef feature) {
    // TODO: Make a default location? currSrcLoc?
    unsigned diagID = diags.getCustomDiagID(
        DiagnosticsEngine::Error, "ClangIR code gen Not Yet Implemented: %0");
    return diags.Report(diagID) << feature;
  }

  DiagnosticBuilder errorNYI(SourceRange, llvm::StringRef);

  template <typename T>
  DiagnosticBuilder errorNYI(SourceRange loc, llvm::StringRef feature,
                             const T &name) {
    return errorNYI(loc.getBegin(), feature, name) << loc;
  }

private:
  // An ordered map of canonical GlobalDecls to their mangled names.
  llvm::MapVector<clang::GlobalDecl, llvm::StringRef> mangledDeclNames;
  llvm::StringMap<clang::GlobalDecl, llvm::BumpPtrAllocator> manglings;
};
} // namespace CIRGen

} // namespace clang

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
