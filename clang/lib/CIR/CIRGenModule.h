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

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H

#include "CIRGenTypes.h"
#include "CIRGenValue.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/CIR/IR/CIRAttrs.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace cir {

class CIRGenFunction;
class CIRGenCXXABI;
class TargetCIRGenInfo;

enum ForDefinition_t : bool { NotForDefinition = false, ForDefinition = true };

/// Implementation of a CIR/MLIR emission from Clang AST.
///
/// This will emit operations that are specific to C(++)/ObjC(++) language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class CIRGenModule {
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;

public:
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &CGO,
               clang::DiagnosticsEngine &Diags);

  ~CIRGenModule();

private:
  mutable std::unique_ptr<TargetCIRGenInfo> TheTargetCIRGenInfo;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  mlir::OpBuilder builder;

  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  const clang::LangOptions &langOpts;

  const clang::CodeGenOptions &codeGenOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  clang::DiagnosticsEngine &Diags;

  const clang::TargetInfo &target;

  std::unique_ptr<CIRGenCXXABI> ABI;

  /// Per-module type mapping from clang AST to CIR.
  CIRGenTypes genTypes;

  /// Per-function codegen information. Updated everytime buildCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *CurCGF = nullptr;

  /// -------
  /// Declaring variables
  /// -------

public:
  mlir::ModuleOp getModule() const { return theModule; }
  mlir::OpBuilder &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() const { return astCtx; }
  const clang::TargetInfo &getTarget() const { return target; }
  const clang::CodeGenOptions &getCodeGenOpts() const { return codeGenOpts; }
  clang::DiagnosticsEngine &getDiags() const { return Diags; }
  CIRGenTypes &getTypes() { return genTypes; }
  const clang::LangOptions &getLangOpts() const { return langOpts; }

  CIRGenCXXABI &getCXXABI() const { return *ABI; }

  // TODO: this obviously overlaps with
  const TargetCIRGenInfo &getTargetCIRGenInfo();

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation SLoc);

  mlir::Location getLoc(clang::SourceRange SLoc);

  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  /// Determine whether an object of this type can be emitted
  /// as a constant.
  ///
  /// If ExcludeCtor is true, the duration when the object's constructor runs
  /// will not be considered. The caller will need to verify that the object is
  /// not written to during its construction.
  /// FIXME: in LLVM codegen path this is part of CGM, which doesn't seem
  /// like necessary, since (1) it doesn't use CGM at all and (2) is AST type
  /// query specific.
  bool isTypeConstant(clang::QualType Ty, bool ExcludeCtor);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// Return the best known alignment for an unknown pointer to a
  /// particular class.
  clang::CharUnits getClassPointerAlignment(const clang::CXXRecordDecl *RD);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  clang::CharUnits getNaturalPointeeTypeAlignment(clang::QualType T,
                                                  LValueBaseInfo *BaseInfo);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  clang::CharUnits getNaturalTypeAlignment(clang::QualType T,
                                           LValueBaseInfo *BaseInfo = nullptr,
                                           bool forPointeeType = false);

  /// A queue of (optional) vtables to consider emitting.
  std::vector<const clang::CXXRecordDecl *> DeferredVTables;

  /// This contains all the decls which have definitions but which are deferred
  /// for emission and therefore should only be output if they are actually
  /// used. If a decl is in this, then it is known to have not been referenced
  /// yet.
  std::map<llvm::StringRef, clang::GlobalDecl> DeferredDecls;

  // This is a list of deferred decls which we have seen that *are* actually
  // referenced. These get code generated when the module is done.
  std::vector<clang::GlobalDecl> DeferredDeclsToEmit;
  void addDeferredDeclToEmit(clang::GlobalDecl GD) {
    DeferredDeclsToEmit.emplace_back(GD);
  }

  void buildTopLevelDecl(clang::Decl *decl);

  /// Emit code for a single global function or var decl. Forward declarations
  /// are emitted lazily.
  void buildGlobal(clang::GlobalDecl D);

  mlir::Type getCIRType(const clang::QualType &type);

  /// Determine whether the definition must be emitted; if this returns \c
  /// false, the definition can be emitted lazily if it's used.
  bool MustBeEmitted(const clang::ValueDecl *D);

  /// Determine whether the definition can be emitted eagerly, or should be
  /// delayed until the end of the translation unit. This is relevant for
  /// definitions whose linkage can change, e.g. implicit function instantions
  /// which may later be explicitly instantiated.
  bool MayBeEmittedEagerly(const clang::ValueDecl *D);

  void verifyModule();

  /// Return the address of the given function. If Ty is non-null, then this
  /// function will use the specified type if it has to create it.
  // TODO: this is a bit weird as `GetAddr` given we give back a FuncOp?
  mlir::FuncOp
  GetAddrOfFunction(clang::GlobalDecl GD, mlir::Type Ty = nullptr,
                    bool ForVTable = false, bool Dontdefer = false,
                    ForDefinition_t IsForDefinition = NotForDefinition);

  llvm::StringRef getMangledName(clang::GlobalDecl GD);

  mlir::Value GetGlobalValue(const clang::Decl *D);

  mlir::Operation *GetGlobalValue(llvm::StringRef Ref);

  // Make sure that this type is translated.
  void UpdateCompletedType(const clang::TagDecl *TD);

  /// Stored a deferred empty coverage mapping for an unused and thus
  /// uninstrumented top level declaration.
  void AddDeferredUnusedCoverageMapping(clang::Decl *D);

  std::nullptr_t getModuleDebugInfo() { return nullptr; }

  /// Emit any needed decls for which code generation was deferred.
  void buildDeferred();

  // Finalize CIR code generation.
  void Release();

  void emitError(const llvm::Twine &message) { theModule.emitError(message); }

private:
  // TODO: CodeGen also passes an AttributeList here. We'll have to match that
  // in CIR
  mlir::FuncOp
  GetOrCreateCIRFunction(llvm::StringRef MangledName, mlir::Type Ty,
                         clang::GlobalDecl D, bool ForVTable,
                         bool DontDefer = false, bool IsThunk = false,
                         ForDefinition_t IsForDefinition = NotForDefinition);

  // An ordered map of canonical GlobalDecls to their mangled names.
  llvm::MapVector<clang::GlobalDecl, llvm::StringRef> MangledDeclNames;
  llvm::StringMap<clang::GlobalDecl, llvm::BumpPtrAllocator> Manglings;
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
