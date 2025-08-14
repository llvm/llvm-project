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
#include "CIRGenCall.h"
#include "CIRGenTypeCache.h"
#include "CIRGenTypes.h"
#include "CIRGenVTables.h"
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

  /// Holds information about C++ vtables.
  CIRGenVTables vtables;

  /// Per-function codegen information. Updated everytime emitCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *curCGF = nullptr;

  llvm::SmallVector<mlir::Attribute> globalScopeAsm;

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

  mlir::Operation *lastGlobalOp = nullptr;

  llvm::DenseMap<const Decl *, cir::GlobalOp> staticLocalDeclMap;

  mlir::Operation *getGlobalValue(llvm::StringRef ref);

  cir::GlobalOp getStaticLocalDeclAddress(const VarDecl *d) {
    return staticLocalDeclMap[d];
  }

  void setStaticLocalDeclAddress(const VarDecl *d, cir::GlobalOp c) {
    staticLocalDeclMap[d] = c;
  }

  cir::GlobalOp getOrCreateStaticVarDecl(const VarDecl &d,
                                         cir::GlobalLinkageKind linkage);

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

  CharUnits computeNonVirtualBaseClassOffset(
      const CXXRecordDecl *derivedClass,
      llvm::iterator_range<CastExpr::path_const_iterator> path);

  /// Get the CIR attributes and calling convention to use for a particular
  /// function type.
  ///
  /// \param calleeInfo - The callee information these attributes are being
  /// constructed for. If valid, the attributes applied to this decl may
  /// contribute to the function attributes and calling convention.
  void constructAttributeList(CIRGenCalleeInfo calleeInfo,
                              mlir::NamedAttrList &attrs);

  /// Will return a global variable of the given type. If a variable with a
  /// different type already exists then a new variable with the right type
  /// will be created and all uses of the old variable will be replaced with a
  /// bitcast to the new variable.
  cir::GlobalOp createOrReplaceCXXRuntimeVariable(
      mlir::Location loc, llvm::StringRef name, mlir::Type ty,
      cir::GlobalLinkageKind linkage, clang::CharUnits alignment);

  /// Return a constant array for the given string.
  mlir::Attribute getConstantArrayFromStringLiteral(const StringLiteral *e);

  /// Return a global symbol reference to a constant array for the given string
  /// literal.
  cir::GlobalOp getGlobalForStringLiteral(const StringLiteral *s,
                                          llvm::StringRef name = ".str");

  /// Set attributes which are common to any form of a global definition (alias,
  /// Objective-C method, function, global variable).
  ///
  /// NOTE: This should only be called for definitions.
  void setCommonAttributes(GlobalDecl gd, mlir::Operation *op);

  const TargetCIRGenInfo &getTargetCIRGenInfo();

  /// Helpers to convert the presumed location of Clang's SourceLocation to an
  /// MLIR Location.
  mlir::Location getLoc(clang::SourceLocation cLoc);
  mlir::Location getLoc(clang::SourceRange cRange);

  /// Return the best known alignment for an unknown pointer to a
  /// particular class.
  clang::CharUnits getClassPointerAlignment(const clang::CXXRecordDecl *rd);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  clang::CharUnits getNaturalTypeAlignment(clang::QualType t,
                                           LValueBaseInfo *baseInfo);

  cir::FuncOp
  getAddrOfCXXStructor(clang::GlobalDecl gd,
                       const CIRGenFunctionInfo *fnInfo = nullptr,
                       cir::FuncType fnType = nullptr, bool dontDefer = false,
                       ForDefinition_t isForDefinition = NotForDefinition) {
    return getAddrAndTypeOfCXXStructor(gd, fnInfo, fnType, dontDefer,
                                       isForDefinition)
        .second;
  }

  std::pair<cir::FuncType, cir::FuncOp> getAddrAndTypeOfCXXStructor(
      clang::GlobalDecl gd, const CIRGenFunctionInfo *fnInfo = nullptr,
      cir::FuncType fnType = nullptr, bool dontDefer = false,
      ForDefinition_t isForDefinition = NotForDefinition);

  mlir::Type getVTableComponentType();
  CIRGenVTables &getVTables() { return vtables; }

  ItaniumVTableContext &getItaniumVTableContext() {
    return vtables.getItaniumVTableContext();
  }
  const ItaniumVTableContext &getItaniumVTableContext() const {
    return vtables.getItaniumVTableContext();
  }

  /// This contains all the decls which have definitions but which are deferred
  /// for emission and therefore should only be output if they are actually
  /// used. If a decl is in this, then it is known to have not been referenced
  /// yet.
  std::map<llvm::StringRef, clang::GlobalDecl> deferredDecls;

  // This is a list of deferred decls which we have seen that *are* actually
  // referenced. These get code generated when the module is done.
  std::vector<clang::GlobalDecl> deferredDeclsToEmit;
  void addDeferredDeclToEmit(clang::GlobalDecl GD) {
    deferredDeclsToEmit.emplace_back(GD);
  }

  void emitTopLevelDecl(clang::Decl *decl);

  /// Determine whether the definition must be emitted; if this returns \c
  /// false, the definition can be emitted lazily if it's used.
  bool mustBeEmitted(const clang::ValueDecl *d);

  /// Determine whether the definition can be emitted eagerly, or should be
  /// delayed until the end of the translation unit. This is relevant for
  /// definitions whose linkage can change, e.g. implicit function
  /// instantiations which may later be explicitly instantiated.
  bool mayBeEmittedEagerly(const clang::ValueDecl *d);

  bool verifyModule() const;

  /// Return the address of the given function. If funcType is non-null, then
  /// this function will use the specified type if it has to create it.
  // TODO: this is a bit weird as `GetAddr` given we give back a FuncOp?
  cir::FuncOp
  getAddrOfFunction(clang::GlobalDecl gd, mlir::Type funcType = nullptr,
                    bool forVTable = false, bool dontDefer = false,
                    ForDefinition_t isForDefinition = NotForDefinition);

  mlir::Operation *
  getAddrOfGlobal(clang::GlobalDecl gd,
                  ForDefinition_t isForDefinition = NotForDefinition);

  /// Emit type info if type of an expression is a variably modified
  /// type. Also emit proper debug info for cast types.
  void emitExplicitCastExprType(const ExplicitCastExpr *e,
                                CIRGenFunction *cgf = nullptr);

  /// Emit code for a single global function or variable declaration. Forward
  /// declarations are emitted lazily.
  void emitGlobal(clang::GlobalDecl gd);

  void emitAliasForGlobal(llvm::StringRef mangledName, mlir::Operation *op,
                          GlobalDecl aliasGD, cir::FuncOp aliasee,
                          cir::GlobalLinkageKind linkage);

  mlir::Type convertType(clang::QualType type);

  /// Set the visibility for the given global.
  void setGlobalVisibility(mlir::Operation *op, const NamedDecl *d) const;
  void setDSOLocal(mlir::Operation *op) const;
  void setDSOLocal(cir::CIRGlobalValueInterface gv) const;

  /// Set visibility, dllimport/dllexport and dso_local.
  /// This must be called after dllimport/dllexport is set.
  void setGVProperties(mlir::Operation *op, const NamedDecl *d) const;
  void setGVPropertiesAux(mlir::Operation *op, const NamedDecl *d) const;

  /// Set function attributes for a function declaration.
  void setFunctionAttributes(GlobalDecl gd, cir::FuncOp f,
                             bool isIncompleteFunction, bool isThunk);

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

  void emitTentativeDefinition(const VarDecl *d);

  // Make sure that this type is translated.
  void updateCompletedType(const clang::TagDecl *td);

  // Produce code for this constructor/destructor. This method doesn't try to
  // apply any ABI rules about which other constructors/destructors are needed
  // or if they are alias to each other.
  cir::FuncOp codegenCXXStructor(clang::GlobalDecl gd);

  bool supportsCOMDAT() const;
  void maybeSetTrivialComdat(const clang::Decl &d, mlir::Operation *op);

  static void setInitializer(cir::GlobalOp &op, mlir::Attribute value);

  void replaceUsesOfNonProtoTypeWithRealFunction(mlir::Operation *old,
                                                 cir::FuncOp newFn);

  cir::FuncOp
  getOrCreateCIRFunction(llvm::StringRef mangledName, mlir::Type funcType,
                         clang::GlobalDecl gd, bool forVTable,
                         bool dontDefer = false, bool isThunk = false,
                         ForDefinition_t isForDefinition = NotForDefinition,
                         mlir::ArrayAttr extraAttrs = {});

  cir::FuncOp createCIRFunction(mlir::Location loc, llvm::StringRef name,
                                cir::FuncType funcType,
                                const clang::FunctionDecl *funcDecl);

  /// Given a builtin id for a function like "__builtin_fabsf", return a
  /// Function* for "fabsf".
  cir::FuncOp getBuiltinLibFunction(const FunctionDecl *fd, unsigned builtinID);

  mlir::IntegerAttr getSize(CharUnits size) {
    return builder.getSizeFromCharUnits(size);
  }

  /// Emit any needed decls for which code generation was deferred.
  void emitDeferred();

  /// Helper for `emitDeferred` to apply actual codegen.
  void emitGlobalDecl(const clang::GlobalDecl &d);

  const llvm::Triple &getTriple() const { return target.getTriple(); }

  // Finalize CIR code generation.
  void release();

  /// -------
  /// Visibility and Linkage
  /// -------

  static mlir::SymbolTable::Visibility
  getMLIRVisibilityFromCIRLinkage(cir::GlobalLinkageKind GLK);
  static cir::VisibilityKind getGlobalVisibilityKindFromClangVisibility(
      clang::VisibilityAttr::VisibilityType visibility);
  cir::VisibilityAttr getGlobalVisibilityAttrFromDecl(const Decl *decl);
  cir::GlobalLinkageKind getFunctionLinkage(GlobalDecl gd);
  static mlir::SymbolTable::Visibility getMLIRVisibility(cir::GlobalOp op);
  cir::GlobalLinkageKind getCIRLinkageForDeclarator(const DeclaratorDecl *dd,
                                                    GVALinkage linkage,
                                                    bool isConstantVariable);
  void setFunctionLinkage(GlobalDecl gd, cir::FuncOp f) {
    cir::GlobalLinkageKind l = getFunctionLinkage(gd);
    f.setLinkageAttr(cir::GlobalLinkageKindAttr::get(&getMLIRContext(), l));
    mlir::SymbolTable::setSymbolVisibility(f,
                                           getMLIRVisibilityFromCIRLinkage(l));
  }

  cir::GlobalLinkageKind getCIRLinkageVarDefinition(const VarDecl *vd,
                                                    bool isConstant);

  void addReplacement(llvm::StringRef name, mlir::Operation *op);

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

  DiagnosticBuilder errorNYI(llvm::StringRef feature) const {
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

  // FIXME: should we use llvm::TrackingVH<mlir::Operation> here?
  typedef llvm::StringMap<mlir::Operation *> ReplacementsTy;
  ReplacementsTy replacements;
  /// Call replaceAllUsesWith on all pairs in replacements.
  void applyReplacements();

  /// A helper function to replace all uses of OldF to NewF that replace
  /// the type of pointer arguments. This is not needed to tradtional
  /// pipeline since LLVM has opaque pointers but CIR not.
  void replacePointerTypeArgs(cir::FuncOp oldF, cir::FuncOp newF);

  void setNonAliasAttributes(GlobalDecl gd, mlir::Operation *op);
};
} // namespace CIRGen

} // namespace clang

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENMODULE_H
