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

#include "CIRDataLayout.h"
#include "CIRGenBuilder.h"
#include "CIRGenTypeCache.h"
#include "CIRGenTypes.h"
#include "CIRGenVTables.h"
#include "CIRGenValue.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

using namespace clang;
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
class CIRGenModule : public CIRGenTypeCache {
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;

public:
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &CGO,
               clang::DiagnosticsEngine &Diags);

  ~CIRGenModule();

  const std::string &getModuleNameHash() const { return ModuleNameHash; }

private:
  mutable std::unique_ptr<TargetCIRGenInfo> TheTargetCIRGenInfo;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  CIRGenBuilderTy builder;

  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  const clang::LangOptions &langOpts;

  const clang::CodeGenOptions &codeGenOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  clang::DiagnosticsEngine &Diags;

  const clang::TargetInfo &target;

  std::unique_ptr<CIRGenCXXABI> ABI;

  /// Used for `UniqueInternalLinkageNames` option
  std::string ModuleNameHash = "";

  /// Per-module type mapping from clang AST to CIR.
  CIRGenTypes genTypes;

  /// Holds information about C++ vtables.
  CIRGenVTables VTables;

  /// Per-function codegen information. Updated everytime buildCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *CurCGF = nullptr;

  // A set of references that have only been set via a weakref so far. This is
  // used to remove the weak of the reference if we ever see a direct reference
  // or a definition.
  llvm::SmallPtrSet<mlir::Operation *, 10> WeakRefReferences;

  /// -------
  /// Declaring variables
  /// -------

  /// Set of global decls for which we already diagnosed mangled name conflict.
  /// Required to not issue a warning (on a mangling conflict) multiple times
  /// for the same decl.
  llvm::DenseSet<clang::GlobalDecl> DiagnosedConflictingDefinitions;

public:
  mlir::ModuleOp getModule() const { return theModule; }
  CIRGenBuilderTy &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() const { return astCtx; }
  const clang::TargetInfo &getTarget() const { return target; }
  const clang::CodeGenOptions &getCodeGenOpts() const { return codeGenOpts; }
  clang::DiagnosticsEngine &getDiags() const { return Diags; }
  CIRGenTypes &getTypes() { return genTypes; }
  const clang::LangOptions &getLangOpts() const { return langOpts; }
  CIRGenFunction *getCurrCIRGenFun() const { return CurCGF; }
  const CIRDataLayout getDataLayout() const {
    // FIXME(cir): instead of creating a CIRDataLayout every time, set it as an
    // attribute for the CIRModule class.
    return {theModule};
  }

  CIRGenCXXABI &getCXXABI() const { return *ABI; }

  /// -------
  /// Handling globals
  /// -------

  // TODO(cir): does this really need to be a state for CIR emission?
  GlobalDecl initializedGlobalDecl;

  /// Global variables with initializers that need to run before main.
  /// TODO(cir): for now track a generation operation, this is so far only
  /// used to sync with DelayedCXXInitPosition. Improve it when we actually
  /// use function calls for initialization
  std::vector<mlir::Operation *> CXXGlobalInits;

  /// Emit the function that initializes C++ globals.
  void buildCXXGlobalInitFunc();

  /// When a C++ decl with an initializer is deferred, null is
  /// appended to CXXGlobalInits, and the index of that null is placed
  /// here so that the initializer will be performed in the correct
  /// order. Once the decl is emitted, the index is replaced with ~0U to ensure
  /// that we don't re-emit the initializer.
  llvm::DenseMap<const Decl *, unsigned> DelayedCXXInitPosition;

  /// Keep track of a map between lambda fields and names, this needs to be per
  /// module since lambdas might get generated later as part of defered work,
  /// and since the pointers are supposed to be uniqued, should be fine. Revisit
  /// this if it ends up taking too much memory.
  llvm::DenseMap<const clang::FieldDecl *, llvm::StringRef> LambdaFieldToName;

  /// If the declaration has internal linkage but is inside an
  /// extern "C" linkage specification, prepare to emit an alias for it
  /// to the expected name.
  template <typename SomeDecl>
  void maybeHandleStaticInExternC(const SomeDecl *D, mlir::cir::GlobalOp GV);

  /// Tell the consumer that this variable has been instantiated.
  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD);

  llvm::DenseMap<const Decl *, mlir::cir::GlobalOp> StaticLocalDeclMap;
  llvm::DenseMap<StringRef, mlir::Value> Globals;
  mlir::Operation *getGlobalValue(StringRef Ref);
  mlir::Value getGlobalValue(const clang::Decl *D);

  /// If the specified mangled name is not in the module, create and return an
  /// mlir::GlobalOp value
  mlir::cir::GlobalOp
  getOrCreateCIRGlobal(StringRef MangledName, mlir::Type Ty, LangAS AddrSpace,
                       const VarDecl *D,
                       ForDefinition_t IsForDefinition = NotForDefinition);

  mlir::cir::GlobalOp getStaticLocalDeclAddress(const VarDecl *D) {
    return StaticLocalDeclMap[D];
  }

  void setStaticLocalDeclAddress(const VarDecl *D, mlir::cir::GlobalOp C) {
    StaticLocalDeclMap[D] = C;
  }

  mlir::cir::GlobalOp
  getOrCreateStaticVarDecl(const VarDecl &D,
                              mlir::cir::GlobalLinkageKind Linkage);

  mlir::cir::GlobalOp buildGlobal(const VarDecl *D, mlir::Type Ty,
                                  ForDefinition_t IsForDefinition);

  /// TODO(cir): once we have cir.module, add this as a convenience method
  /// there instead of here.
  ///
  /// Look up the specified global in the module symbol table.
  ///   1. If it does not exist, add a declaration of the global and return it.
  ///   2. Else, the global exists but has the wrong type: return the function
  ///      with a constantexpr cast to the right type.
  ///   3. Finally, if the existing global is the correct declaration, return
  ///      the existing global.
  mlir::cir::GlobalOp getOrInsertGlobal(
      mlir::Location loc, StringRef Name, mlir::Type Ty,
      llvm::function_ref<mlir::cir::GlobalOp()> CreateGlobalCallback);

  // Overload to construct a global variable using its constructor's defaults.
  mlir::cir::GlobalOp getOrInsertGlobal(mlir::Location loc, StringRef Name,
                                        mlir::Type Ty);

  static mlir::cir::GlobalOp createGlobalOp(CIRGenModule &CGM,
                                            mlir::Location loc, StringRef name,
                                            mlir::Type t, bool isCst = false);

  /// Return the mlir::Value for the address of the given global variable.
  /// If Ty is non-null and if the global doesn't exist, then it will be created
  /// with the specified type instead of whatever the normal requested type
  /// would be. If IsForDefinition is true, it is guaranteed that an actual
  /// global with type Ty will be returned, not conversion of a variable with
  /// the same mangled name but some other type.
  mlir::Value
  getAddrOfGlobalVar(const VarDecl *D, mlir::Type Ty = {},
                     ForDefinition_t IsForDefinition = NotForDefinition);

  /// Return the mlir::GlobalViewAttr for the address of the given global.
  mlir::cir::GlobalViewAttr
  getAddrOfGlobalVarAttr(const VarDecl *D, mlir::Type Ty = {},
                         ForDefinition_t IsForDefinition = NotForDefinition);

  /// Get a reference to the target of VD.
  mlir::Operation* getWeakRefReference(const ValueDecl *VD);

  CharUnits
  computeNonVirtualBaseClassOffset(const CXXRecordDecl *DerivedClass,
                                   CastExpr::path_const_iterator Start,
                                   CastExpr::path_const_iterator End);

  /// Will return a global variable of the given type. If a variable with a
  /// different type already exists then a new variable with the right type
  /// will be created and all uses of the old variable will be replaced with a
  /// bitcast to the new variable.
  mlir::cir::GlobalOp createOrReplaceCXXRuntimeVariable(
      mlir::Location loc, StringRef Name, mlir::Type Ty,
      mlir::cir::GlobalLinkageKind Linkage, clang::CharUnits Alignment);

  /// Emit any vtables which we deferred and still have a use for.
  void buildDeferredVTables();
  bool shouldOpportunisticallyEmitVTables();

  /// Return the appropriate linkage for the vtable, VTT, and type information
  /// of the given class.
  mlir::cir::GlobalLinkageKind getVTableLinkage(const CXXRecordDecl *RD);

  /// Emit type metadata for the given vtable using the given layout.
  void buildVTableTypeMetadata(const CXXRecordDecl *RD,
                               mlir::cir::GlobalOp VTable,
                               const VTableLayout &VTLayout);

  /// Get the address of the RTTI descriptor for the given type.
  mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc, QualType Ty,
                                          bool ForEH = false);

  /// TODO(cir): add CIR visibility bits.
  static mlir::SymbolTable::Visibility getCIRVisibility(Visibility V) {
    switch (V) {
    case DefaultVisibility:
      return mlir::SymbolTable::Visibility::Public;
    case HiddenVisibility:
      return mlir::SymbolTable::Visibility::Private;
    case ProtectedVisibility:
      llvm_unreachable("NYI");
    }
    llvm_unreachable("unknown visibility!");
  }

  llvm::DenseMap<mlir::Attribute, mlir::cir::GlobalOp> ConstantStringMap;

  /// Return a constant array for the given string.
  mlir::Attribute getConstantArrayFromStringLiteral(const StringLiteral *E);

  /// Return a global symbol reference to a constant array for the given string
  /// literal.
  mlir::SymbolRefAttr
  getAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                     StringRef Name = ".str");
  unsigned StringLiteralCnt = 0;

  /// Return the AST address space of constant literal, which is used to emit
  /// the constant literal as global variable in LLVM IR.
  /// Note: This is not necessarily the address space of the constant literal
  /// in AST. For address space agnostic language, e.g. C++, constant literal
  /// in AST is always in default address space.
  LangAS getGlobalConstantAddressSpace() const;

  /// Set attributes which are common to any form of a global definition (alias,
  /// Objective-C method, function, global variable).
  ///
  /// NOTE: This should only be called for definitions.
  void setCommonAttributes(GlobalDecl GD, mlir::Operation *GV);

  // TODO: this obviously overlaps with
  const TargetCIRGenInfo &getTargetCIRGenInfo();

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation SLoc);
  mlir::Location getLoc(clang::SourceRange SLoc);
  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  /// Helper to convert Clang's alignment to CIR alignment
  mlir::IntegerAttr getSize(CharUnits size);

  /// Returns whether the given record has public LTO visibility (regardless of
  /// -lto-whole-program-visibility) and therefore may not participate in
  /// (single-module) CFI and whole-program vtable optimization.
  bool AlwaysHasLTOVisibilityPublic(const CXXRecordDecl *RD);

  /// Returns whether the given record has hidden LTO visibility and therefore
  /// may participate in (single-module) CFI and whole-program vtable
  /// optimization.
  bool HasHiddenLTOVisibility(const CXXRecordDecl *RD);

  /// Determine whether an object of this type can be emitted
  /// as a constant.
  ///
  /// If ExcludeCtor is true, the duration when the object's constructor runs
  /// will not be considered. The caller will need to verify that the object is
  /// not written to during its construction.
  /// FIXME: in LLVM codegen path this is part of CGM, which doesn't seem
  /// like necessary, since (1) it doesn't use CGM at all and (2) is AST type
  /// query specific.
  bool isTypeConstant(clang::QualType Ty, bool ExcludeCtor, bool ExcludeDtor);

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

  mlir::cir::FuncOp getAddrOfCXXStructor(
      clang::GlobalDecl GD, const CIRGenFunctionInfo *FnInfo = nullptr,
      mlir::cir::FuncType FnType = nullptr, bool DontDefer = false,
      ForDefinition_t IsForDefinition = NotForDefinition) {

    return getAddrAndTypeOfCXXStructor(GD, FnInfo, FnType, DontDefer,
                                       IsForDefinition)
        .second;
  }

  /// A queue of (optional) vtables to consider emitting.
  std::vector<const clang::CXXRecordDecl *> DeferredVTables;

  mlir::Type getVTableComponentType();
  CIRGenVTables &getVTables() { return VTables; }

  ItaniumVTableContext &getItaniumVTableContext() {
    return VTables.getItaniumVTableContext();
  }
  const ItaniumVTableContext &getItaniumVTableContext() const {
    return VTables.getItaniumVTableContext();
  }

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

  // After HandleTranslation finishes, differently from DeferredDeclsToEmit,
  // DefaultMethodsToEmit is only called after a set of CIR passes run. See
  // addDefaultMethodsToEmit usage for examples.
  std::vector<clang::GlobalDecl> DefaultMethodsToEmit;
  void addDefaultMethodsToEmit(clang::GlobalDecl GD) {
    DefaultMethodsToEmit.emplace_back(GD);
  }

  std::pair<mlir::cir::FuncType, mlir::cir::FuncOp> getAddrAndTypeOfCXXStructor(
      clang::GlobalDecl GD, const CIRGenFunctionInfo *FnInfo = nullptr,
      mlir::cir::FuncType FnType = nullptr, bool Dontdefer = false,
      ForDefinition_t IsForDefinition = NotForDefinition);

  void buildTopLevelDecl(clang::Decl *decl);
  void buildLinkageSpec(const LinkageSpecDecl *D);

  /// Emit code for a single global function or var decl. Forward declarations
  /// are emitted lazily.
  void buildGlobal(clang::GlobalDecl D);

  mlir::Type getCIRType(const clang::QualType &type);

  /// Set the visibility for the given global.
  void setGlobalVisibility(mlir::Operation *Op, const NamedDecl *D) const;
  void setDSOLocal(mlir::Operation *Op) const;
  /// Set visibility, dllimport/dllexport and dso_local.
  /// This must be called after dllimport/dllexport is set.
  void setGVProperties(mlir::Operation *Op, const NamedDecl *D) const;
  void setGVPropertiesAux(mlir::Operation *Op, const NamedDecl *D) const;

  /// Determine whether the definition must be emitted; if this returns \c
  /// false, the definition can be emitted lazily if it's used.
  bool MustBeEmitted(const clang::ValueDecl *D);

  /// Whether this function's return type has no side effects, and thus may be
  /// trivially discared if it is unused.
  bool MayDropFunctionReturn(const clang::ASTContext &Context,
                             clang::QualType ReturnType);

  bool isInNoSanitizeList(clang::SanitizerMask Kind, mlir::cir::FuncOp Fn,
                          clang::SourceLocation) const;

  /// Determine whether the definition can be emitted eagerly, or should be
  /// delayed until the end of the translation unit. This is relevant for
  /// definitions whose linkage can change, e.g. implicit function instantions
  /// which may later be explicitly instantiated.
  bool MayBeEmittedEagerly(const clang::ValueDecl *D);

  bool verifyModule();

  /// Return the address of the given function. If Ty is non-null, then this
  /// function will use the specified type if it has to create it.
  // TODO: this is a bit weird as `GetAddr` given we give back a FuncOp?
  mlir::cir::FuncOp
  GetAddrOfFunction(clang::GlobalDecl GD, mlir::Type Ty = nullptr,
                    bool ForVTable = false, bool Dontdefer = false,
                    ForDefinition_t IsForDefinition = NotForDefinition);

  mlir::Operation *
  GetAddrOfGlobal(clang::GlobalDecl GD,
                  ForDefinition_t IsForDefinition = NotForDefinition);

  // C++ related functions.
  void buildDeclContext(const DeclContext *DC);

  /// Return the result of value-initializing the given type, i.e. a null
  /// expression of the given type.  This is usually, but not always, an LLVM
  /// null constant.
  mlir::Value buildNullConstant(QualType T, mlir::Location loc);

  llvm::StringRef getMangledName(clang::GlobalDecl GD);

  void buildTentativeDefinition(const VarDecl *D);

  // Make sure that this type is translated.
  void UpdateCompletedType(const clang::TagDecl *TD);

  void buildGlobalDefinition(clang::GlobalDecl D,
                             mlir::Operation *Op = nullptr);
  void buildGlobalFunctionDefinition(clang::GlobalDecl D, mlir::Operation *Op);
  void buildGlobalVarDefinition(const clang::VarDecl *D,
                                bool IsTentative = false);

  /// Emit the function that initializes the specified global
  void buildGlobalVarDeclInit(const VarDecl *D, mlir::cir::GlobalOp Addr,
                                  bool PerformInit);

  void addDeferredVTable(const CXXRecordDecl *RD) {
    DeferredVTables.push_back(RD);
  }

  /// Stored a deferred empty coverage mapping for an unused and thus
  /// uninstrumented top level declaration.
  void AddDeferredUnusedCoverageMapping(clang::Decl *D);

  std::nullptr_t getModuleDebugInfo() { return nullptr; }

  /// Emit any needed decls for which code generation was deferred.
  void buildDeferred(unsigned recursionLimit);

  /// Helper for `buildDeferred` to apply actual codegen.
  void buildGlobalDecl(clang::GlobalDecl &D);

  /// Build default methods not emitted before this point.
  void buildDefaultMethods();

  const llvm::Triple &getTriple() const { return target.getTriple(); }

  // Finalize CIR code generation.
  void Release();

  bool shouldEmitFunction(clang::GlobalDecl GD);

  // Produce code for this constructor/destructor. This method doesn't try to
  // apply any ABI rules about which other constructors/destructors are needed
  // or if they are alias to each other.
  mlir::cir::FuncOp codegenCXXStructor(clang::GlobalDecl GD);

  // Produce code for this constructor/destructor for global initialzation.
  void codegenGlobalInitCxxStructor(const clang::VarDecl *D,
                                    mlir::cir::GlobalOp Addr, bool NeedsCtor,
                                    bool NeedsDtor);

  bool lookupRepresentativeDecl(llvm::StringRef MangledName,
                                clang::GlobalDecl &Result) const;

  bool supportsCOMDAT() const;
  void maybeSetTrivialComdat(const clang::Decl &D, mlir::Operation *Op);

  void emitError(const llvm::Twine &message) { theModule.emitError(message); }

  /// -------
  /// Visibility and Linkage
  /// -------

  static void setInitializer(mlir::cir::GlobalOp &op, mlir::Attribute value);
  static mlir::SymbolTable::Visibility
  getMLIRVisibilityFromCIRLinkage(mlir::cir::GlobalLinkageKind GLK);
  static mlir::SymbolTable::Visibility
  getMLIRVisibility(mlir::cir::GlobalOp op);
  mlir::cir::GlobalLinkageKind getFunctionLinkage(GlobalDecl GD);
  mlir::cir::GlobalLinkageKind
  getCIRLinkageForDeclarator(const DeclaratorDecl *D, GVALinkage Linkage,
                             bool IsConstantVariable);
  void setFunctionLinkage(GlobalDecl GD, mlir::cir::FuncOp f) {
    auto L = getFunctionLinkage(GD);
    f.setLinkageAttr(
        mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), L));
    mlir::SymbolTable::setSymbolVisibility(f,
                                           getMLIRVisibilityFromCIRLinkage(L));
  }

  mlir::cir::GlobalLinkageKind getCIRLinkageVarDefinition(const VarDecl *VD,
                                                          bool IsConstant);

  void addReplacement(StringRef Name, mlir::Operation *Op);

  mlir::Location getLocForFunction(const clang::FunctionDecl *FD);

  void ReplaceUsesOfNonProtoTypeWithRealFunction(mlir::Operation *Old,
                                                 mlir::cir::FuncOp NewFn);

  void setExtraAttributesForFunc(mlir::cir::FuncOp f,
                                 const clang::FunctionDecl *FD);

  // TODO: CodeGen also passes an AttributeList here. We'll have to match that
  // in CIR
  mlir::cir::FuncOp
  GetOrCreateCIRFunction(llvm::StringRef MangledName, mlir::Type Ty,
                         clang::GlobalDecl D, bool ForVTable,
                         bool DontDefer = false, bool IsThunk = false,
                         ForDefinition_t IsForDefinition = NotForDefinition);
  // Effectively create the CIR instruction, properly handling insertion
  // points.
  mlir::cir::FuncOp createCIRFunction(mlir::Location loc, StringRef name,
                                      mlir::cir::FuncType Ty,
                                      const clang::FunctionDecl *FD);

  /// Emit type info if type of an expression is a variably modified
  /// type. Also emit proper debug info for cast types.
  void buildExplicitCastExprType(const ExplicitCastExpr *E,
                                 CIRGenFunction *CGF = nullptr);

  static constexpr const char *builtinCoroId = "__builtin_coro_id";
  static constexpr const char *builtinCoroAlloc = "__builtin_coro_alloc";
  static constexpr const char *builtinCoroBegin = "__builtin_coro_begin";
  static constexpr const char *builtinCoroEnd = "__builtin_coro_end";

  /// Given a builtin id for a function like "__builtin_fabsf", return a
  /// Function* for "fabsf".
  mlir::cir::FuncOp getBuiltinLibFunction(const FunctionDecl *FD,
                                          unsigned BuiltinID);

  /// Emit a general error that something can't be done.
  void Error(SourceLocation loc, StringRef error);

  /// Print out an error that codegen doesn't support the specified stmt yet.
  void ErrorUnsupported(const Stmt *S, const char *Type);

  /// Print out an error that codegen doesn't support the specified decl yet.
  void ErrorUnsupported(const Decl *D, const char *Type);

private:
  // An ordered map of canonical GlobalDecls to their mangled names.
  llvm::MapVector<clang::GlobalDecl, llvm::StringRef> MangledDeclNames;
  llvm::StringMap<clang::GlobalDecl, llvm::BumpPtrAllocator> Manglings;

  // FIXME: should we use llvm::TrackingVH<mlir::Operation> here?
  typedef llvm::StringMap<mlir::Operation *> ReplacementsTy;
  ReplacementsTy Replacements;
  /// Call replaceAllUsesWith on all pairs in Replacements.
  void applyReplacements();

  /// Map source language used to a CIR attribute.
  mlir::cir::SourceLanguage getCIRSourceLanguage();
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
