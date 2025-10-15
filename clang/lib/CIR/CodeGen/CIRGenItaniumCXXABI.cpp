//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/ExprCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class CIRGenItaniumCXXABI : public CIRGenCXXABI {
protected:
  /// All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, cir::GlobalOp> vtables;

public:
  CIRGenItaniumCXXABI(CIRGenModule &cgm) : CIRGenCXXABI(cgm) {
    assert(!cir::MissingFeatures::cxxabiUseARMMethodPtrABI());
    assert(!cir::MissingFeatures::cxxabiUseARMGuardVarABI());
  }

  AddedStructorArgs getImplicitConstructorArgs(CIRGenFunction &cgf,
                                               const CXXConstructorDecl *d,
                                               CXXCtorType type,
                                               bool forVirtualBase,
                                               bool delegating) override;

  bool needsVTTParameter(clang::GlobalDecl gd) override;

  AddedStructorArgCounts
  buildStructorSignature(GlobalDecl gd,
                         llvm::SmallVectorImpl<CanQualType> &argTys) override;

  void emitInstanceFunctionProlog(SourceLocation loc,
                                  CIRGenFunction &cgf) override;

  void addImplicitStructorParams(CIRGenFunction &cgf, QualType &resTy,
                                 FunctionArgList &params) override;
  mlir::Value getCXXDestructorImplicitParam(CIRGenFunction &cgf,
                                            const CXXDestructorDecl *dd,
                                            CXXDtorType type,
                                            bool forVirtualBase,
                                            bool delegating) override;
  void emitCXXConstructors(const clang::CXXConstructorDecl *d) override;
  void emitCXXDestructors(const clang::CXXDestructorDecl *d) override;
  void emitCXXStructor(clang::GlobalDecl gd) override;

  void emitDestructorCall(CIRGenFunction &cgf, const CXXDestructorDecl *dd,
                          CXXDtorType type, bool forVirtualBase,
                          bool delegating, Address thisAddr,
                          QualType thisTy) override;
  void registerGlobalDtor(const VarDecl *vd, cir::FuncOp dtor,
                          mlir::Value addr) override;

  void emitRethrow(CIRGenFunction &cgf, bool isNoReturn) override;
  void emitThrow(CIRGenFunction &cgf, const CXXThrowExpr *e) override;

  bool useThunkForDtorVariant(const CXXDestructorDecl *dtor,
                              CXXDtorType dt) const override {
    // Itanium does not emit any destructor variant as an inline thunk.
    // Delegating may occur as an optimization, but all variants are either
    // emitted with external linkage or as linkonce if they are inline and used.
    return false;
  }

  bool isVirtualOffsetNeededForVTableField(CIRGenFunction &cgf,
                                           CIRGenFunction::VPtr vptr) override;

  cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *rd,
                                CharUnits vptrOffset) override;
  CIRGenCallee getVirtualFunctionPointer(CIRGenFunction &cgf,
                                         clang::GlobalDecl gd, Address thisAddr,
                                         mlir::Type ty,
                                         SourceLocation loc) override;
  mlir::Value emitVirtualDestructorCall(CIRGenFunction &cgf,
                                        const CXXDestructorDecl *dtor,
                                        CXXDtorType dtorType, Address thisAddr,
                                        DeleteOrMemberCallExpr e) override;
  mlir::Value getVTableAddressPoint(BaseSubobject base,
                                    const CXXRecordDecl *vtableClass) override;
  mlir::Value getVTableAddressPointInStructorWithVTT(
      CIRGenFunction &cgf, const CXXRecordDecl *vtableClass, BaseSubobject base,
      const CXXRecordDecl *nearestVBase);

  mlir::Value getVTableAddressPointInStructor(
      CIRGenFunction &cgf, const clang::CXXRecordDecl *vtableClass,
      clang::BaseSubobject base,
      const clang::CXXRecordDecl *nearestVBase) override;
  void emitVTableDefinitions(CIRGenVTables &cgvt,
                             const CXXRecordDecl *rd) override;
  void emitVirtualInheritanceTables(const CXXRecordDecl *rd) override;

  mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc,
                                          QualType ty) override;

  bool doStructorsInitializeVPtrs(const CXXRecordDecl *vtableClass) override {
    return true;
  }

  mlir::Value
  getVirtualBaseClassOffset(mlir::Location loc, CIRGenFunction &cgf,
                            Address thisAddr, const CXXRecordDecl *classDecl,
                            const CXXRecordDecl *baseClassDecl) override;

  // The traditional clang CodeGen emits calls to `__dynamic_cast` directly into
  // LLVM in the `emitDynamicCastCall` function. In CIR, `dynamic_cast`
  // expressions are lowered to `cir.dyn_cast` ops instead of calls to runtime
  // functions. So during CIRGen we don't need the `emitDynamicCastCall`
  // function that clang CodeGen has.
  mlir::Value emitDynamicCast(CIRGenFunction &cgf, mlir::Location loc,
                              QualType srcRecordTy, QualType destRecordTy,
                              cir::PointerType destCIRTy, bool isRefCast,
                              Address src) override;

  /**************************** RTTI Uniqueness ******************************/
protected:
  /// Returns true if the ABI requires RTTI type_info objects to be unique
  /// across a program.
  virtual bool shouldRTTIBeUnique() const { return true; }

public:
  /// What sort of unique-RTTI behavior should we use?
  enum RTTIUniquenessKind {
    /// We are guaranteeing, or need to guarantee, that the RTTI string
    /// is unique.
    RUK_Unique,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// can demote to hidden visibility but must use string comparisons.
    RUK_NonUniqueHidden,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// have to use string comparisons, but we also have to emit it with
    /// non-hidden visibility.
    RUK_NonUniqueVisible
  };

  /// Return the required visibility status for the given type and linkage in
  /// the current ABI.
  RTTIUniquenessKind
  classifyRTTIUniqueness(QualType canTy, cir::GlobalLinkageKind linkage) const;
};

} // namespace

void CIRGenItaniumCXXABI::emitInstanceFunctionProlog(SourceLocation loc,
                                                     CIRGenFunction &cgf) {
  // Naked functions have no prolog.
  if (cgf.curFuncDecl && cgf.curFuncDecl->hasAttr<NakedAttr>()) {
    cgf.cgm.errorNYI(cgf.curFuncDecl->getLocation(),
                     "emitInstanceFunctionProlog: Naked");
  }

  /// Initialize the 'this' slot. In the Itanium C++ ABI, no prologue
  /// adjustments are required, because they are all handled by thunks.
  setCXXABIThisValue(cgf, loadIncomingCXXThis(cgf));

  /// Initialize the 'vtt' slot if needed.
  if (getStructorImplicitParamDecl(cgf)) {
    cir::LoadOp val = cgf.getBuilder().createLoad(
        cgf.getLoc(loc),
        cgf.getAddrOfLocalVar(getStructorImplicitParamDecl(cgf)));
    setStructorImplicitParamValue(cgf, val);
  }

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CIRGenCXXBI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation
  if (hasThisReturn(cgf.curGD)) {
    cgf.cgm.errorNYI(cgf.curFuncDecl->getLocation(),
                     "emitInstanceFunctionProlog: hasThisReturn");
  }
}

CIRGenCXXABI::AddedStructorArgCounts
CIRGenItaniumCXXABI::buildStructorSignature(
    GlobalDecl gd, llvm::SmallVectorImpl<CanQualType> &argTys) {
  clang::ASTContext &astContext = cgm.getASTContext();

  // All parameters are already in place except VTT, which goes after 'this'.
  // These are clang types, so we don't need to worry about sret yet.

  // Check if we need to add a VTT parameter (which has type void **).
  if ((isa<CXXConstructorDecl>(gd.getDecl()) ? gd.getCtorType() == Ctor_Base
                                             : gd.getDtorType() == Dtor_Base) &&
      cast<CXXMethodDecl>(gd.getDecl())->getParent()->getNumVBases() != 0) {
    assert(!cir::MissingFeatures::addressSpace());
    argTys.insert(argTys.begin() + 1,
                  astContext.getPointerType(
                      CanQualType::CreateUnsafe(astContext.VoidPtrTy)));
    return AddedStructorArgCounts::withPrefix(1);
  }

  return AddedStructorArgCounts{};
}

// Find out how to cirgen the complete destructor and constructor
namespace {
enum class StructorCIRGen { Emit, RAUW, Alias, COMDAT };
}

static StructorCIRGen getCIRGenToUse(CIRGenModule &cgm,
                                     const CXXMethodDecl *md) {
  if (!cgm.getCodeGenOpts().CXXCtorDtorAliases)
    return StructorCIRGen::Emit;

  // The complete and base structors are not equivalent if there are any virtual
  // bases, so emit separate functions.
  if (md->getParent()->getNumVBases())
    return StructorCIRGen::Emit;

  GlobalDecl aliasDecl;
  if (const auto *dd = dyn_cast<CXXDestructorDecl>(md)) {
    aliasDecl = GlobalDecl(dd, Dtor_Complete);
  } else {
    const auto *cd = cast<CXXConstructorDecl>(md);
    aliasDecl = GlobalDecl(cd, Ctor_Complete);
  }

  cir::GlobalLinkageKind linkage = cgm.getFunctionLinkage(aliasDecl);

  if (cir::isDiscardableIfUnused(linkage))
    return StructorCIRGen::RAUW;

  // FIXME: Should we allow available_externally aliases?
  if (!cir::isValidLinkage(linkage))
    return StructorCIRGen::RAUW;

  if (cir::isWeakForLinker(linkage)) {
    // Only ELF and wasm support COMDATs with arbitrary names (C5/D5).
    if (cgm.getTarget().getTriple().isOSBinFormatELF() ||
        cgm.getTarget().getTriple().isOSBinFormatWasm())
      return StructorCIRGen::COMDAT;
    return StructorCIRGen::Emit;
  }

  return StructorCIRGen::Alias;
}

static void emitConstructorDestructorAlias(CIRGenModule &cgm,
                                           GlobalDecl aliasDecl,
                                           GlobalDecl targetDecl) {
  cir::GlobalLinkageKind linkage = cgm.getFunctionLinkage(aliasDecl);

  // Does this function alias already exists?
  StringRef mangledName = cgm.getMangledName(aliasDecl);
  auto globalValue = dyn_cast_or_null<cir::CIRGlobalValueInterface>(
      cgm.getGlobalValue(mangledName));
  if (globalValue && !globalValue.isDeclaration())
    return;

  auto entry = cast_or_null<cir::FuncOp>(cgm.getGlobalValue(mangledName));

  // Retrieve aliasee info.
  auto aliasee = cast<cir::FuncOp>(cgm.getAddrOfGlobal(targetDecl));

  // Populate actual alias.
  cgm.emitAliasForGlobal(mangledName, entry, aliasDecl, aliasee, linkage);
}

void CIRGenItaniumCXXABI::emitCXXStructor(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());
  StructorCIRGen cirGenType = getCIRGenToUse(cgm, md);
  const auto *cd = dyn_cast<CXXConstructorDecl>(md);

  if (cd ? gd.getCtorType() == Ctor_Complete
         : gd.getDtorType() == Dtor_Complete) {
    GlobalDecl baseDecl =
        cd ? gd.getWithCtorType(Ctor_Base) : gd.getWithDtorType(Dtor_Base);
    ;

    if (cirGenType == StructorCIRGen::Alias ||
        cirGenType == StructorCIRGen::COMDAT) {
      emitConstructorDestructorAlias(cgm, gd, baseDecl);
      return;
    }

    if (cirGenType == StructorCIRGen::RAUW) {
      StringRef mangledName = cgm.getMangledName(gd);
      mlir::Operation *aliasee = cgm.getAddrOfGlobal(baseDecl);
      cgm.addReplacement(mangledName, aliasee);
      return;
    }
  }

  auto fn = cgm.codegenCXXStructor(gd);

  cgm.maybeSetTrivialComdat(*md, fn);
}

void CIRGenItaniumCXXABI::addImplicitStructorParams(CIRGenFunction &cgf,
                                                    QualType &resTy,
                                                    FunctionArgList &params) {
  const auto *md = cast<CXXMethodDecl>(cgf.curGD.getDecl());
  assert(isa<CXXConstructorDecl>(md) || isa<CXXDestructorDecl>(md));

  // Check if we need a VTT parameter as well.
  if (needsVTTParameter(cgf.curGD)) {
    ASTContext &astContext = cgm.getASTContext();

    // FIXME: avoid the fake decl
    assert(!cir::MissingFeatures::addressSpace());
    QualType t = astContext.getPointerType(astContext.VoidPtrTy);
    auto *vttDecl = ImplicitParamDecl::Create(
        astContext, /*DC=*/nullptr, md->getLocation(),
        &astContext.Idents.get("vtt"), t, ImplicitParamKind::CXXVTT);
    params.insert(params.begin() + 1, vttDecl);
    getStructorImplicitParamDecl(cgf) = vttDecl;
  }
}

void CIRGenItaniumCXXABI::emitCXXConstructors(const CXXConstructorDecl *d) {
  // Just make sure we're in sync with TargetCXXABI.
  assert(cgm.getTarget().getCXXABI().hasConstructorVariants());

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  cgm.emitGlobal(GlobalDecl(d, Ctor_Base));

  // The constructor used for constructing this as a complete class;
  // constructs the virtual bases, then calls the base constructor.
  if (!d->getParent()->isAbstract()) {
    // We don't need to emit the complete ctro if the class is abstract.
    cgm.emitGlobal(GlobalDecl(d, Ctor_Complete));
  }
}

void CIRGenItaniumCXXABI::emitCXXDestructors(const CXXDestructorDecl *d) {
  // The destructor used for destructing this as a base class; ignores
  // virtual bases.
  cgm.emitGlobal(GlobalDecl(d, Dtor_Base));

  // The destructor used for destructing this as a most-derived class;
  // call the base destructor and then destructs any virtual bases.
  cgm.emitGlobal(GlobalDecl(d, Dtor_Complete));

  // The destructor in a virtual table is always a 'deleting'
  // destructor, which calls the complete destructor and then uses the
  // appropriate operator delete.
  if (d->isVirtual())
    cgm.emitGlobal(GlobalDecl(d, Dtor_Deleting));
}

CIRGenCXXABI::AddedStructorArgs CIRGenItaniumCXXABI::getImplicitConstructorArgs(
    CIRGenFunction &cgf, const CXXConstructorDecl *d, CXXCtorType type,
    bool forVirtualBase, bool delegating) {
  if (!needsVTTParameter(GlobalDecl(d, type)))
    return AddedStructorArgs{};

  // Insert the implicit 'vtt' argument as the second argument. Make sure to
  // correctly reflect its address space, which can differ from generic on
  // some targets.
  mlir::Value vtt =
      cgf.getVTTParameter(GlobalDecl(d, type), forVirtualBase, delegating);
  QualType vttTy =
      cgm.getASTContext().getPointerType(cgm.getASTContext().VoidPtrTy);
  assert(!cir::MissingFeatures::addressSpace());
  return AddedStructorArgs::withPrefix({{vtt, vttTy}});
}

/// Return whether the given global decl needs a VTT (virtual table table)
/// parameter, which it does if it's a base constructor or destructor with
/// virtual bases.
bool CIRGenItaniumCXXABI::needsVTTParameter(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  // We don't have any virtual bases, just return early.
  if (!md->getParent()->getNumVBases())
    return false;

  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(md) && gd.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(md) && gd.getDtorType() == Dtor_Base)
    return true;

  return false;
}

void CIRGenItaniumCXXABI::emitVTableDefinitions(CIRGenVTables &cgvt,
                                                const CXXRecordDecl *rd) {
  cir::GlobalOp vtable = getAddrOfVTable(rd, CharUnits());
  if (vtable.hasInitializer())
    return;

  ItaniumVTableContext &vtContext = cgm.getItaniumVTableContext();
  const VTableLayout &vtLayout = vtContext.getVTableLayout(rd);
  cir::GlobalLinkageKind linkage = cgm.getVTableLinkage(rd);
  mlir::Attribute rtti =
      cgm.getAddrOfRTTIDescriptor(cgm.getLoc(rd->getBeginLoc()),
                                  cgm.getASTContext().getCanonicalTagType(rd));

  // Classic codegen uses ConstantInitBuilder here, which is a very general
  // and feature-rich class to generate initializers for global values.
  // For now, this is using a simpler approach to create the initializer in CIR.
  cgvt.createVTableInitializer(vtable, vtLayout, rtti,
                               cir::isLocalLinkage(linkage));

  // Set the correct linkage.
  vtable.setLinkage(linkage);

  if (cgm.supportsCOMDAT() && cir::isWeakForLinker(linkage))
    vtable.setComdat(true);

  // Set the right visibility.
  cgm.setGVProperties(vtable, rd);

  // If this is the magic class __cxxabiv1::__fundamental_type_info,
  // we will emit the typeinfo for the fundamental types. This is the
  // same behaviour as GCC.
  const DeclContext *DC = rd->getDeclContext();
  if (rd->getIdentifier() &&
      rd->getIdentifier()->isStr("__fundamental_type_info") &&
      isa<NamespaceDecl>(DC) && cast<NamespaceDecl>(DC)->getIdentifier() &&
      cast<NamespaceDecl>(DC)->getIdentifier()->isStr("__cxxabiv1") &&
      DC->getParent()->isTranslationUnit()) {
    cgm.errorNYI(rd->getSourceRange(),
                 "emitVTableDefinitions: __fundamental_type_info");
  }

  auto vtableAsGlobalValue = dyn_cast<cir::CIRGlobalValueInterface>(*vtable);
  assert(vtableAsGlobalValue && "VTable must support CIRGlobalValueInterface");
  // Always emit type metadata on non-available_externally definitions, and on
  // available_externally definitions if we are performing whole program
  // devirtualization. For WPD we need the type metadata on all vtable
  // definitions to ensure we associate derived classes with base classes
  // defined in headers but with a strong definition only in a shared
  // library.
  assert(!cir::MissingFeatures::vtableEmitMetadata());
  if (cgm.getCodeGenOpts().WholeProgramVTables) {
    cgm.errorNYI(rd->getSourceRange(),
                 "emitVTableDefinitions: WholeProgramVTables");
  }

  assert(!cir::MissingFeatures::vtableRelativeLayout());
  if (vtContext.isRelativeLayout()) {
    cgm.errorNYI(rd->getSourceRange(), "vtableRelativeLayout");
  }
}

mlir::Value CIRGenItaniumCXXABI::emitVirtualDestructorCall(
    CIRGenFunction &cgf, const CXXDestructorDecl *dtor, CXXDtorType dtorType,
    Address thisAddr, DeleteOrMemberCallExpr expr) {
  auto *callExpr = dyn_cast<const CXXMemberCallExpr *>(expr);
  auto *delExpr = dyn_cast<const CXXDeleteExpr *>(expr);
  assert((callExpr != nullptr) ^ (delExpr != nullptr));
  assert(callExpr == nullptr || callExpr->arg_begin() == callExpr->arg_end());
  assert(dtorType == Dtor_Deleting || dtorType == Dtor_Complete);

  GlobalDecl globalDecl(dtor, dtorType);
  const CIRGenFunctionInfo *fnInfo =
      &cgm.getTypes().arrangeCXXStructorDeclaration(globalDecl);
  const cir::FuncType &fnTy = cgm.getTypes().getFunctionType(*fnInfo);
  auto callee = CIRGenCallee::forVirtual(callExpr, globalDecl, thisAddr, fnTy);

  QualType thisTy =
      callExpr ? callExpr->getObjectType() : delExpr->getDestroyedType();

  cgf.emitCXXDestructorCall(globalDecl, callee, thisAddr.emitRawPointer(),
                            thisTy, nullptr, QualType(), nullptr);
  return nullptr;
}

void CIRGenItaniumCXXABI::emitVirtualInheritanceTables(
    const CXXRecordDecl *rd) {
  CIRGenVTables &vtables = cgm.getVTables();
  cir::GlobalOp vtt = vtables.getAddrOfVTT(rd);
  vtables.emitVTTDefinition(vtt, cgm.getVTableLinkage(rd), rd);
}

namespace {
class CIRGenItaniumRTTIBuilder {
  CIRGenModule &cgm;                 // Per-module state.
  const CIRGenItaniumCXXABI &cxxABI; // Per-module state.

  /// The fields of the RTTI descriptor currently being built.
  SmallVector<mlir::Attribute, 16> fields;

  // Returns the mangled type name of the given type.
  cir::GlobalOp getAddrOfTypeName(mlir::Location loc, QualType ty,
                                  cir::GlobalLinkageKind linkage);

  /// descriptor of the given type.
  mlir::Attribute getAddrOfExternalRTTIDescriptor(mlir::Location loc,
                                                  QualType ty);

  /// Build the vtable pointer for the given type.
  void buildVTablePointer(mlir::Location loc, const Type *ty);

  /// Build an abi::__si_class_type_info, used for single inheritance, according
  /// to the Itanium C++ ABI, 2.9.5p6b.
  void buildSIClassTypeInfo(mlir::Location loc, const CXXRecordDecl *rd);

  /// Build an abi::__vmi_class_type_info, used for
  /// classes with bases that do not satisfy the abi::__si_class_type_info
  /// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
  void buildVMIClassTypeInfo(mlir::Location loc, const CXXRecordDecl *rd);

public:
  CIRGenItaniumRTTIBuilder(const CIRGenItaniumCXXABI &abi, CIRGenModule &cgm)
      : cgm(cgm), cxxABI(abi) {}

  /// Build the RTTI type info struct for the given type, or
  /// link to an existing RTTI descriptor if one already exists.
  mlir::Attribute buildTypeInfo(mlir::Location loc, QualType ty);

  /// Build the RTTI type info struct for the given type.
  mlir::Attribute buildTypeInfo(mlir::Location loc, QualType ty,
                                cir::GlobalLinkageKind linkage,
                                mlir::SymbolTable::Visibility visibility);
};
} // namespace

// TODO(cir): Will be removed after sharing them with the classical codegen
namespace {

// Pointer type info flags.
enum {
  /// PTI_Const - Type has const qualifier.
  PTI_Const = 0x1,

  /// PTI_Volatile - Type has volatile qualifier.
  PTI_Volatile = 0x2,

  /// PTI_Restrict - Type has restrict qualifier.
  PTI_Restrict = 0x4,

  /// PTI_Incomplete - Type is incomplete.
  PTI_Incomplete = 0x8,

  /// PTI_ContainingClassIncomplete - Containing class is incomplete.
  /// (in pointer to member).
  PTI_ContainingClassIncomplete = 0x10,

  /// PTI_TransactionSafe - Pointee is transaction_safe function (C++ TM TS).
  // PTI_TransactionSafe = 0x20,

  /// PTI_Noexcept - Pointee is noexcept function (C++1z).
  PTI_Noexcept = 0x40,
};

// VMI type info flags.
enum {
  /// VMI_NonDiamondRepeat - Class has non-diamond repeated inheritance.
  VMI_NonDiamondRepeat = 0x1,

  /// VMI_DiamondShaped - Class is diamond shaped.
  VMI_DiamondShaped = 0x2
};

// Base class type info flags.
enum {
  /// BCTI_Virtual - Base class is virtual.
  BCTI_Virtual = 0x1,

  /// BCTI_Public - Base class is public.
  BCTI_Public = 0x2
};

/// Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
/// TODO(cir): this can unified with LLVM codegen
static bool typeInfoIsInStandardLibrary(const BuiltinType *ty) {
  // Itanium C++ ABI 2.9.2:
  //   Basic type information (e.g. for "int", "bool", etc.) will be kept in
  //   the run-time support library. Specifically, the run-time support
  //   library should contain type_info objects for the types X, X* and
  //   X const*, for every X in: void, std::nullptr_t, bool, wchar_t, char,
  //   unsigned char, signed char, short, unsigned short, int, unsigned int,
  //   long, unsigned long, long long, unsigned long long, float, double,
  //   long double, char16_t, char32_t, and the IEEE 754r decimal and
  //   half-precision floating point types.
  //
  // GCC also emits RTTI for __int128.
  // FIXME: We do not emit RTTI information for decimal types here.

  // Types added here must also be added to emitFundamentalRTTIDescriptors.
  switch (ty->getKind()) {
  case BuiltinType::WasmExternRef:
  case BuiltinType::HLSLResource:
    llvm_unreachable("NYI");
  case BuiltinType::Void:
  case BuiltinType::NullPtr:
  case BuiltinType::Bool:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char_U:
  case BuiltinType::Char_S:
  case BuiltinType::UChar:
  case BuiltinType::SChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::Half:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float16:
  case BuiltinType::Float128:
  case BuiltinType::Ibm128:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return true;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLClkEvent:
  case BuiltinType::OCLQueue:
  case BuiltinType::OCLReserveID:
#define SVE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/AArch64ACLETypes.def"
#define PPC_VECTOR_TYPE(Name, Id, Size) case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
#define AMDGPU_TYPE(Name, Id, SingletonId, Width, Align) case BuiltinType::Id:
#include "clang/Basic/AMDGPUTypes.def"
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
  case BuiltinType::BFloat16:
    return false;

  case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
    llvm_unreachable("asking for RRTI for a placeholder type!");

  case BuiltinType::ObjCId:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCSel:
    llvm_unreachable("FIXME: Objective-C types are unsupported!");
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

static bool typeInfoIsInStandardLibrary(const PointerType *pointerTy) {
  QualType pointeeTy = pointerTy->getPointeeType();
  const auto *builtinTy = dyn_cast<BuiltinType>(pointeeTy);
  if (!builtinTy)
    return false;

  // Check the qualifiers.
  Qualifiers quals = pointeeTy.getQualifiers();
  quals.removeConst();

  if (!quals.empty())
    return false;

  return typeInfoIsInStandardLibrary(builtinTy);
}

/// IsStandardLibraryRTTIDescriptor - Returns whether the type
/// information for the given type exists in the standard library.
static bool isStandardLibraryRttiDescriptor(QualType ty) {
  // Type info for builtin types is defined in the standard library.
  if (const auto *builtinTy = dyn_cast<BuiltinType>(ty))
    return typeInfoIsInStandardLibrary(builtinTy);

  // Type info for some pointer types to builtin types is defined in the
  // standard library.
  if (const auto *pointerTy = dyn_cast<PointerType>(ty))
    return typeInfoIsInStandardLibrary(pointerTy);

  return false;
}

/// ShouldUseExternalRTTIDescriptor - Returns whether the type information for
/// the given type exists somewhere else, and that we should not emit the type
/// information in this translation unit.  Assumes that it is not a
/// standard-library type.
static bool shouldUseExternalRttiDescriptor(CIRGenModule &cgm, QualType ty) {
  ASTContext &context = cgm.getASTContext();

  // If RTTI is disabled, assume it might be disabled in the
  // translation unit that defines any potential key function, too.
  if (!context.getLangOpts().RTTI)
    return false;

  if (const auto *recordTy = dyn_cast<RecordType>(ty)) {
    const CXXRecordDecl *rd =
        cast<CXXRecordDecl>(recordTy->getOriginalDecl())->getDefinitionOrSelf();
    if (!rd->hasDefinition())
      return false;

    if (!rd->isDynamicClass())
      return false;

    // FIXME: this may need to be reconsidered if the key function
    // changes.
    // N.B. We must always emit the RTTI data ourselves if there exists a key
    // function.
    bool isDLLImport = rd->hasAttr<DLLImportAttr>();

    // Don't import the RTTI but emit it locally.
    if (cgm.getTriple().isOSCygMing())
      return false;

    if (cgm.getVTables().isVTableExternal(rd)) {
      if (cgm.getTarget().hasPS4DLLImportExport())
        return true;

      return !isDLLImport || cgm.getTriple().isWindowsItaniumEnvironment();
    }

    if (isDLLImport)
      return true;
  }

  return false;
}

/// Contains virtual and non-virtual bases seen when traversing a class
/// hierarchy.
struct SeenBases {
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> nonVirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> virtualBases;
};

/// Compute the value of the flags member in abi::__vmi_class_type_info.
///
static unsigned computeVmiClassTypeInfoFlags(const CXXBaseSpecifier *base,
                                             SeenBases &bases) {

  unsigned flags = 0;
  auto *baseDecl = base->getType()->castAsCXXRecordDecl();

  if (base->isVirtual()) {
    // Mark the virtual base as seen.
    if (!bases.virtualBases.insert(baseDecl).second) {
      // If this virtual base has been seen before, then the class is diamond
      // shaped.
      flags |= VMI_DiamondShaped;
    } else {
      if (bases.nonVirtualBases.count(baseDecl))
        flags |= VMI_NonDiamondRepeat;
    }
  } else {
    // Mark the non-virtual base as seen.
    if (!bases.nonVirtualBases.insert(baseDecl).second) {
      // If this non-virtual base has been seen before, then the class has non-
      // diamond shaped repeated inheritance.
      flags |= VMI_NonDiamondRepeat;
    } else {
      if (bases.virtualBases.count(baseDecl))
        flags |= VMI_NonDiamondRepeat;
    }
  }

  // Walk all bases.
  for (const auto &bs : baseDecl->bases())
    flags |= computeVmiClassTypeInfoFlags(&bs, bases);

  return flags;
}

static unsigned computeVmiClassTypeInfoFlags(const CXXRecordDecl *rd) {
  unsigned flags = 0;
  SeenBases bases;

  // Walk all bases.
  for (const auto &bs : rd->bases())
    flags |= computeVmiClassTypeInfoFlags(&bs, bases);

  return flags;
}

// Return whether the given record decl has a "single,
// public, non-virtual base at offset zero (i.e. the derived class is dynamic
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
// TODO(cir): this can unified with LLVM codegen
static bool canUseSingleInheritance(const CXXRecordDecl *rd) {
  // Check the number of bases.
  if (rd->getNumBases() != 1)
    return false;

  // Get the base.
  CXXRecordDecl::base_class_const_iterator base = rd->bases_begin();

  // Check that the base is not virtual.
  if (base->isVirtual())
    return false;

  // Check that the base is public.
  if (base->getAccessSpecifier() != AS_public)
    return false;

  // Check that the class is dynamic iff the base is.
  auto *baseDecl = base->getType()->castAsCXXRecordDecl();
  return baseDecl->isEmpty() ||
         baseDecl->isDynamicClass() == rd->isDynamicClass();
}

/// IsIncompleteClassType - Returns whether the given record type is incomplete.
static bool isIncompleteClassType(const RecordType *recordTy) {
  return !recordTy->getOriginalDecl()
              ->getDefinitionOrSelf()
              ->isCompleteDefinition();
}

/// Returns whether the given type contains an
/// incomplete class type. This is true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whoise pointee type contains an
///     incomplete class type.
/// is an indirect or direct pointer to an incomplete class type.
static bool containsIncompleteClassType(QualType ty) {
  if (const auto *recordTy = dyn_cast<RecordType>(ty)) {
    if (isIncompleteClassType(recordTy))
      return true;
  }

  if (const auto *pointerTy = dyn_cast<PointerType>(ty))
    return containsIncompleteClassType(pointerTy->getPointeeType());

  if (const auto *memberPointerTy = dyn_cast<MemberPointerType>(ty)) {
    // Check if the class type is incomplete.
    if (!memberPointerTy->getMostRecentCXXRecordDecl()->hasDefinition())
      return true;

    return containsIncompleteClassType(memberPointerTy->getPointeeType());
  }

  return false;
}

const char *vTableClassNameForType(const CIRGenModule &cgm, const Type *ty) {
  // abi::__class_type_info.
  static const char *const classTypeInfo =
      "_ZTVN10__cxxabiv117__class_type_infoE";
  // abi::__si_class_type_info.
  static const char *const siClassTypeInfo =
      "_ZTVN10__cxxabiv120__si_class_type_infoE";
  // abi::__vmi_class_type_info.
  static const char *const vmiClassTypeInfo =
      "_ZTVN10__cxxabiv121__vmi_class_type_infoE";

  switch (ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    llvm_unreachable("Pipe types shouldn't get here");

  case Type::ArrayParameter:
    llvm_unreachable("Array Parameter types should not get here.");

  case Type::Builtin:
  case Type::BitInt:
  // GCC treats vector and complex types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::Atomic:
  // FIXME: GCC treats block pointers as fundamental types?!
  case Type::BlockPointer:
    return "_ZTVN10__cxxabiv123__fundamental_type_infoE";
  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    cgm.errorNYI("VTableClassNameForType: __array_type_info");
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    cgm.errorNYI("VTableClassNameForType: __function_type_info");
    break;

  case Type::Enum:
    cgm.errorNYI("VTableClassNameForType: Enum");
    break;

  case Type::Record: {
    const CXXRecordDecl *rd =
        cast<CXXRecordDecl>(cast<RecordType>(ty)->getOriginalDecl())
            ->getDefinitionOrSelf();

    if (!rd->hasDefinition() || !rd->getNumBases()) {
      return classTypeInfo;
    }

    if (canUseSingleInheritance(rd)) {
      return siClassTypeInfo;
    }

    return vmiClassTypeInfo;
  }

  case Type::ObjCObject:
    cgm.errorNYI("VTableClassNameForType: ObjCObject");
    break;

  case Type::ObjCInterface:
    cgm.errorNYI("VTableClassNameForType: ObjCInterface");
    break;

  case Type::ObjCObjectPointer:
  case Type::Pointer:
    cgm.errorNYI("VTableClassNameForType: __pointer_type_info");
    break;

  case Type::MemberPointer:
    cgm.errorNYI("VTableClassNameForType: __pointer_to_member_type_info");
    break;

  case Type::HLSLAttributedResource:
  case Type::HLSLInlineSpirv:
    llvm_unreachable("HLSL doesn't support virtual functions");
  }

  return nullptr;
}
} // namespace

/// Return the linkage that the type info and type info name constants
/// should have for the given type.
static cir::GlobalLinkageKind getTypeInfoLinkage(CIRGenModule &cgm,
                                                 QualType ty) {
  //   In addition, it and all of the intermediate abi::__pointer_type_info
  //   structs in the chain down to the abi::__class_type_info for the
  //   incomplete class type must be prevented from resolving to the
  //   corresponding type_info structs for the complete class type, possibly
  //   by making them local static objects. Finally, a dummy class RTTI is
  //   generated for the incomplete type that will not resolve to the final
  //   complete class RTTI (because the latter need not exist), possibly by
  //   making it a local static object.
  if (containsIncompleteClassType(ty))
    return cir::GlobalLinkageKind::InternalLinkage;

  switch (ty->getLinkage()) {
  case Linkage::Invalid:
    llvm_unreachable("Linkage hasn't been computed!");

  case Linkage::None:
  case Linkage::Internal:
  case Linkage::UniqueExternal:
    return cir::GlobalLinkageKind::InternalLinkage;

  case Linkage::VisibleNone:
  case Linkage::Module:
  case Linkage::External:
    // RTTI is not enabled, which means that this type info struct is going
    // to be used for exception handling. Give it linkonce_odr linkage.
    if (!cgm.getLangOpts().RTTI)
      return cir::GlobalLinkageKind::LinkOnceODRLinkage;

    if (const RecordType *record = dyn_cast<RecordType>(ty)) {
      const CXXRecordDecl *rd =
          cast<CXXRecordDecl>(record->getOriginalDecl())->getDefinitionOrSelf();
      if (rd->hasAttr<WeakAttr>())
        return cir::GlobalLinkageKind::WeakODRLinkage;

      if (cgm.getTriple().isWindowsItaniumEnvironment())
        if (rd->hasAttr<DLLImportAttr>() &&
            shouldUseExternalRttiDescriptor(cgm, ty))
          return cir::GlobalLinkageKind::ExternalLinkage;

      // MinGW always uses LinkOnceODRLinkage for type info.
      if (rd->isDynamicClass() && !cgm.getASTContext()
                                       .getTargetInfo()
                                       .getTriple()
                                       .isWindowsGNUEnvironment())
        return cgm.getVTableLinkage(rd);
    }

    return cir::GlobalLinkageKind::LinkOnceODRLinkage;
  }

  llvm_unreachable("Invalid linkage!");
}

cir::GlobalOp
CIRGenItaniumRTTIBuilder::getAddrOfTypeName(mlir::Location loc, QualType ty,
                                            cir::GlobalLinkageKind linkage) {
  CIRGenBuilderTy &builder = cgm.getBuilder();
  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTIName(ty, out);

  // We know that the mangled name of the type starts at index 4 of the
  // mangled name of the typename, so we can just index into it in order to
  // get the mangled name of the type.
  mlir::Attribute init = builder.getString(
      name.substr(4), cgm.convertType(cgm.getASTContext().CharTy),
      std::nullopt);

  CharUnits align =
      cgm.getASTContext().getTypeAlignInChars(cgm.getASTContext().CharTy);

  // builder.getString can return a #cir.zero if the string given to it only
  // contains null bytes. However, type names cannot be full of null bytes.
  // So cast Init to a ConstArrayAttr should be safe.
  auto initStr = cast<cir::ConstArrayAttr>(init);

  cir::GlobalOp gv = cgm.createOrReplaceCXXRuntimeVariable(
      loc, name, initStr.getType(), linkage, align);
  CIRGenModule::setInitializer(gv, init);
  return gv;
}

mlir::Attribute
CIRGenItaniumRTTIBuilder::getAddrOfExternalRTTIDescriptor(mlir::Location loc,
                                                          QualType ty) {
  // Mangle the RTTI name.
  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTI(ty, out);
  CIRGenBuilderTy &builder = cgm.getBuilder();

  // Look for an existing global.
  cir::GlobalOp gv = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(cgm.getModule(), name));

  if (!gv) {
    // Create a new global variable.
    // From LLVM codegen => Note for the future: If we would ever like to do
    // deferred emission of RTTI, check if emitting vtables opportunistically
    // need any adjustment.
    gv = CIRGenModule::createGlobalOp(cgm, loc, name, builder.getUInt8PtrTy(),
                                      /*isConstant=*/true);
    const CXXRecordDecl *rd = ty->getAsCXXRecordDecl();
    cgm.setGVProperties(gv, rd);

    // Import the typeinfo symbol when all non-inline virtual methods are
    // imported.
    if (cgm.getTarget().hasPS4DLLImportExport()) {
      cgm.errorNYI("getAddrOfExternalRTTIDescriptor: hasPS4DLLImportExport");
    }
  }

  return builder.getGlobalViewAttr(builder.getUInt8PtrTy(), gv);
}

void CIRGenItaniumRTTIBuilder::buildVTablePointer(mlir::Location loc,
                                                  const Type *ty) {
  CIRGenBuilderTy &builder = cgm.getBuilder();
  const char *vTableName = vTableClassNameForType(cgm, ty);

  // Check if the alias exists. If it doesn't, then get or create the global.
  if (cgm.getItaniumVTableContext().isRelativeLayout()) {
    cgm.errorNYI("buildVTablePointer: isRelativeLayout");
    return;
  }

  mlir::Type vtableGlobalTy = builder.getPointerTo(builder.getUInt8PtrTy());
  llvm::Align align = cgm.getDataLayout().getABITypeAlign(vtableGlobalTy);
  cir::GlobalOp vTable = cgm.createOrReplaceCXXRuntimeVariable(
      loc, vTableName, vtableGlobalTy, cir::GlobalLinkageKind::ExternalLinkage,
      CharUnits::fromQuantity(align));

  // The vtable address point is 2.
  mlir::Attribute field{};
  if (cgm.getItaniumVTableContext().isRelativeLayout()) {
    cgm.errorNYI("buildVTablePointer: isRelativeLayout");
  } else {
    SmallVector<mlir::Attribute, 4> offsets{
        cgm.getBuilder().getI32IntegerAttr(2)};
    auto indices = mlir::ArrayAttr::get(builder.getContext(), offsets);
    field = cgm.getBuilder().getGlobalViewAttr(cgm.getBuilder().getUInt8PtrTy(),
                                               vTable, indices);
  }

  assert(field && "expected attribute");
  fields.push_back(field);
}

/// Build an abi::__si_class_type_info, used for single inheritance, according
/// to the Itanium C++ ABI, 2.95p6b.
void CIRGenItaniumRTTIBuilder::buildSIClassTypeInfo(mlir::Location loc,
                                                    const CXXRecordDecl *rd) {
  // Itanium C++ ABI 2.9.5p6b:
  // It adds to abi::__class_type_info a single member pointing to the
  // type_info structure for the base type,
  mlir::Attribute baseTypeInfo =
      CIRGenItaniumRTTIBuilder(cxxABI, cgm)
          .buildTypeInfo(loc, rd->bases_begin()->getType());
  fields.push_back(baseTypeInfo);
}

/// Build an abi::__vmi_class_type_info, used for
/// classes with bases that do not satisfy the abi::__si_class_type_info
/// constraints, according to the Itanium C++ ABI, 2.9.5p5c.
void CIRGenItaniumRTTIBuilder::buildVMIClassTypeInfo(mlir::Location loc,
                                                     const CXXRecordDecl *rd) {
  mlir::Type unsignedIntLTy =
      cgm.convertType(cgm.getASTContext().UnsignedIntTy);

  // Itanium C++ ABI 2.9.5p6c:
  //   __flags is a word with flags describing details about the class
  //   structure, which may be referenced by using the __flags_masks
  //   enumeration. These flags refer to both direct and indirect bases.
  unsigned flags = computeVmiClassTypeInfoFlags(rd);
  fields.push_back(cir::IntAttr::get(unsignedIntLTy, flags));

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_count is a word with the number of direct proper base class
  //   descriptions that follow.
  fields.push_back(cir::IntAttr::get(unsignedIntLTy, rd->getNumBases()));

  if (!rd->getNumBases())
    return;

  // Now add the base class descriptions.

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_info[] is an array of base class descriptions -- one for every
  //   direct proper base. Each description is of the type:
  //
  //   struct abi::__base_class_type_info {
  //   public:
  //     const __class_type_info *__base_type;
  //     long __offset_flags;
  //
  //     enum __offset_flags_masks {
  //       __virtual_mask = 0x1,
  //       __public_mask = 0x2,
  //       __offset_shift = 8
  //     };
  //   };

  // If we're in mingw and 'long' isn't wide enough for a pointer, use 'long
  // long' instead of 'long' for __offset_flags. libstdc++abi uses long long on
  // LLP64 platforms.
  // FIXME: Consider updating libc++abi to match, and extend this logic to all
  // LLP64 platforms.
  QualType offsetFlagsTy = cgm.getASTContext().LongTy;
  const TargetInfo &ti = cgm.getASTContext().getTargetInfo();
  if (ti.getTriple().isOSCygMing() &&
      ti.getPointerWidth(LangAS::Default) > ti.getLongWidth())
    offsetFlagsTy = cgm.getASTContext().LongLongTy;
  mlir::Type offsetFlagsLTy = cgm.convertType(offsetFlagsTy);

  for (const CXXBaseSpecifier &base : rd->bases()) {
    // The __base_type member points to the RTTI for the base type.
    fields.push_back(CIRGenItaniumRTTIBuilder(cxxABI, cgm)
                         .buildTypeInfo(loc, base.getType()));

    CXXRecordDecl *baseDecl = base.getType()->castAsCXXRecordDecl();
    int64_t offsetFlags = 0;

    // All but the lower 8 bits of __offset_flags are a signed offset.
    // For a non-virtual base, this is the offset in the object of the base
    // subobject. For a virtual base, this is the offset in the virtual table of
    // the virtual base offset for the virtual base referenced (negative).
    CharUnits offset;
    if (base.isVirtual())
      offset = cgm.getItaniumVTableContext().getVirtualBaseOffsetOffset(
          rd, baseDecl);
    else {
      const ASTRecordLayout &layout =
          cgm.getASTContext().getASTRecordLayout(rd);
      offset = layout.getBaseClassOffset(baseDecl);
    }
    offsetFlags = uint64_t(offset.getQuantity()) << 8;

    // The low-order byte of __offset_flags contains flags, as given by the
    // masks from the enumeration __offset_flags_masks.
    if (base.isVirtual())
      offsetFlags |= BCTI_Virtual;
    if (base.getAccessSpecifier() == AS_public)
      offsetFlags |= BCTI_Public;

    fields.push_back(cir::IntAttr::get(offsetFlagsLTy, offsetFlags));
  }
}

mlir::Attribute CIRGenItaniumRTTIBuilder::buildTypeInfo(mlir::Location loc,
                                                        QualType ty) {
  // We want to operate on the canonical type.
  ty = ty.getCanonicalType();

  // Check if we've already emitted an RTTI descriptor for this type.
  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTI(ty, out);

  auto oldGV = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(cgm.getModule(), name));

  if (oldGV && !oldGV.isDeclaration()) {
    assert(!oldGV.hasAvailableExternallyLinkage() &&
           "available_externally typeinfos not yet implemented");
    return cgm.getBuilder().getGlobalViewAttr(cgm.getBuilder().getUInt8PtrTy(),
                                              oldGV);
  }

  // Check if there is already an external RTTI descriptor for this type.
  if (isStandardLibraryRttiDescriptor(ty) ||
      shouldUseExternalRttiDescriptor(cgm, ty))
    return getAddrOfExternalRTTIDescriptor(loc, ty);

  // Emit the standard library with external linkage.
  cir::GlobalLinkageKind linkage = getTypeInfoLinkage(cgm, ty);

  // Give the type_info object and name the formal visibility of the
  // type itself.
  assert(!cir::MissingFeatures::hiddenVisibility());
  assert(!cir::MissingFeatures::protectedVisibility());

  mlir::SymbolTable::Visibility symVisibility;
  if (cir::isLocalLinkage(linkage))
    // If the linkage is local, only default visibility makes sense.
    symVisibility = mlir::SymbolTable::Visibility::Public;
  else if (cxxABI.classifyRTTIUniqueness(ty, linkage) ==
           CIRGenItaniumCXXABI::RUK_NonUniqueHidden) {
    cgm.errorNYI(
        "buildTypeInfo: classifyRTTIUniqueness == RUK_NonUniqueHidden");
    symVisibility = CIRGenModule::getMLIRVisibility(ty->getVisibility());
  } else
    symVisibility = CIRGenModule::getMLIRVisibility(ty->getVisibility());

  return buildTypeInfo(loc, ty, linkage, symVisibility);
}

mlir::Attribute CIRGenItaniumRTTIBuilder::buildTypeInfo(
    mlir::Location loc, QualType ty, cir::GlobalLinkageKind linkage,
    mlir::SymbolTable::Visibility visibility) {
  CIRGenBuilderTy &builder = cgm.getBuilder();

  assert(!cir::MissingFeatures::setDLLStorageClass());

  // Add the vtable pointer.
  buildVTablePointer(loc, cast<Type>(ty));

  // And the name.
  cir::GlobalOp typeName = getAddrOfTypeName(loc, ty, linkage);
  mlir::Attribute typeNameField;

  // If we're supposed to demote the visibility, be sure to set a flag
  // to use a string comparison for type_info comparisons.
  CIRGenItaniumCXXABI::RTTIUniquenessKind rttiUniqueness =
      cxxABI.classifyRTTIUniqueness(ty, linkage);
  if (rttiUniqueness != CIRGenItaniumCXXABI::RUK_Unique) {
    // The flag is the sign bit, which on ARM64 is defined to be clear
    // for global pointers. This is very ARM64-specific.
    cgm.errorNYI(
        "buildTypeInfo: rttiUniqueness != CIRGenItaniumCXXABI::RUK_Unique");
  } else {
    typeNameField =
        builder.getGlobalViewAttr(builder.getUInt8PtrTy(), typeName);
  }

  fields.push_back(typeNameField);

  switch (ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  // GCC treats vector types as fundamental types.
  case Type::Builtin:
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::BlockPointer:
    // Itanium C++ ABI 2.9.5p4:
    // abi::__fundamental_type_info adds no data members to std::type_info.
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    break;

  case Type::BitInt:
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::ArrayParameter:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__array_type_info adds no data members to std::type_info.
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__function_type_info adds no data members to std::type_info.
    break;

  case Type::Enum:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__enum_type_info adds no data members to std::type_info.
    break;

  case Type::Record: {
    const auto *rd =
        cast<CXXRecordDecl>(cast<RecordType>(ty)->getOriginalDecl())
            ->getDefinitionOrSelf();
    if (!rd->hasDefinition() || !rd->getNumBases()) {
      // We don't need to emit any fields.
      break;
    }

    if (canUseSingleInheritance(rd)) {
      buildSIClassTypeInfo(loc, rd);
    } else {
      buildVMIClassTypeInfo(loc, rd);
    }

    break;
  }

  case Type::ObjCObject:
  case Type::ObjCInterface:
    cgm.errorNYI("buildTypeInfo: ObjCObject & ObjCInterface");
    break;

  case Type::ObjCObjectPointer:
    cgm.errorNYI("buildTypeInfo: ObjCObjectPointer");
    break;

  case Type::Pointer:
    cgm.errorNYI("buildTypeInfo: Pointer");
    break;

  case Type::MemberPointer:
    cgm.errorNYI("buildTypeInfo: MemberPointer");
    break;

  case Type::Atomic:
    // No fields, at least for the moment.
    break;

  case Type::HLSLAttributedResource:
  case Type::HLSLInlineSpirv:
    llvm_unreachable("HLSL doesn't support RTTI");
  }

  assert(!cir::MissingFeatures::opGlobalDLLImportExport());
  cir::TypeInfoAttr init = builder.getTypeInfo(builder.getArrayAttr(fields));

  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTI(ty, out);

  // Create new global and search for an existing global.
  auto oldGV = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(cgm.getModule(), name));

  cir::GlobalOp gv =
      CIRGenModule::createGlobalOp(cgm, loc, name, init.getType(),
                                   /*isConstant=*/true);

  // Export the typeinfo in the same circumstances as the vtable is
  // exported.
  if (cgm.getTarget().hasPS4DLLImportExport()) {
    cgm.errorNYI("buildTypeInfo: target hasPS4DLLImportExport");
    return {};
  }

  // If there's already an old global variable, replace it with the new one.
  if (oldGV) {
    // Replace occurrences of the old variable if needed.
    gv.setName(oldGV.getName());
    if (!oldGV->use_empty()) {
      cgm.errorNYI("buildTypeInfo: old GV !use_empty");
      return {};
    }
    oldGV->erase();
  }

  if (cgm.supportsCOMDAT() && cir::isWeakForLinker(gv.getLinkage())) {
    assert(!cir::MissingFeatures::setComdat());
    cgm.errorNYI("buildTypeInfo: supportsCOMDAT & isWeakForLinker");
    return {};
  }

  CharUnits align = cgm.getASTContext().toCharUnitsFromBits(
      cgm.getTarget().getPointerAlign(LangAS::Default));
  gv.setAlignmentAttr(cgm.getSize(align));

  // The Itanium ABI specifies that type_info objects must be globally
  // unique, with one exception: if the type is an incomplete class
  // type or a (possibly indirect) pointer to one.  That exception
  // affects the general case of comparing type_info objects produced
  // by the typeid operator, which is why the comparison operators on
  // std::type_info generally use the type_info name pointers instead
  // of the object addresses.  However, the language's built-in uses
  // of RTTI generally require class types to be complete, even when
  // manipulating pointers to those class types.  This allows the
  // implementation of dynamic_cast to rely on address equality tests,
  // which is much faster.

  // All of this is to say that it's important that both the type_info
  // object and the type_info name be uniqued when weakly emitted.

  mlir::SymbolTable::setSymbolVisibility(typeName, visibility);
  assert(!cir::MissingFeatures::setDLLStorageClass());
  assert(!cir::MissingFeatures::opGlobalPartition());
  assert(!cir::MissingFeatures::setDSOLocal());

  mlir::SymbolTable::setSymbolVisibility(gv, visibility);
  assert(!cir::MissingFeatures::setDLLStorageClass());
  assert(!cir::MissingFeatures::opGlobalPartition());
  assert(!cir::MissingFeatures::setDSOLocal());

  CIRGenModule::setInitializer(gv, init);
  return builder.getGlobalViewAttr(builder.getUInt8PtrTy(), gv);
}

mlir::Attribute CIRGenItaniumCXXABI::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                             QualType ty) {
  return CIRGenItaniumRTTIBuilder(*this, cgm).buildTypeInfo(loc, ty);
}

/// What sort of uniqueness rules should we use for the RTTI for the
/// given type?
CIRGenItaniumCXXABI::RTTIUniquenessKind
CIRGenItaniumCXXABI::classifyRTTIUniqueness(
    QualType canTy, cir::GlobalLinkageKind linkage) const {
  if (shouldRTTIBeUnique())
    return RUK_Unique;

  // It's only necessary for linkonce_odr or weak_odr linkage.
  if (linkage != cir::GlobalLinkageKind::LinkOnceODRLinkage &&
      linkage != cir::GlobalLinkageKind::WeakODRLinkage)
    return RUK_Unique;

  // It's only necessary with default visibility.
  if (canTy->getVisibility() != DefaultVisibility)
    return RUK_Unique;

  // If we're not required to publish this symbol, hide it.
  if (linkage == cir::GlobalLinkageKind::LinkOnceODRLinkage)
    return RUK_NonUniqueHidden;

  // If we're required to publish this symbol, as we might be under an
  // explicit instantiation, leave it with default visibility but
  // enable string-comparisons.
  assert(linkage == cir::GlobalLinkageKind::WeakODRLinkage);
  return RUK_NonUniqueVisible;
}

void CIRGenItaniumCXXABI::emitDestructorCall(
    CIRGenFunction &cgf, const CXXDestructorDecl *dd, CXXDtorType type,
    bool forVirtualBase, bool delegating, Address thisAddr, QualType thisTy) {
  GlobalDecl gd(dd, type);
  mlir::Value vtt =
      getCXXDestructorImplicitParam(cgf, dd, type, forVirtualBase, delegating);
  ASTContext &astContext = cgm.getASTContext();
  QualType vttTy = astContext.getPointerType(astContext.VoidPtrTy);
  assert(!cir::MissingFeatures::appleKext());
  CIRGenCallee callee =
      CIRGenCallee::forDirect(cgm.getAddrOfCXXStructor(gd), gd);

  cgf.emitCXXDestructorCall(gd, callee, thisAddr.getPointer(), thisTy, vtt,
                            vttTy, nullptr);
}

void CIRGenItaniumCXXABI::registerGlobalDtor(const VarDecl *vd,
                                             cir::FuncOp dtor,
                                             mlir::Value addr) {
  if (vd->isNoDestroy(cgm.getASTContext()))
    return;

  if (vd->getTLSKind()) {
    cgm.errorNYI(vd->getSourceRange(), "registerGlobalDtor: TLS");
    return;
  }

  // HLSL doesn't support atexit.
  if (cgm.getLangOpts().HLSL) {
    cgm.errorNYI(vd->getSourceRange(), "registerGlobalDtor: HLSL");
    return;
  }

  // The default behavior is to use atexit. This is handled in lowering
  // prepare. Nothing to be done for CIR here.
}

mlir::Value CIRGenItaniumCXXABI::getCXXDestructorImplicitParam(
    CIRGenFunction &cgf, const CXXDestructorDecl *dd, CXXDtorType type,
    bool forVirtualBase, bool delegating) {
  GlobalDecl gd(dd, type);
  return cgf.getVTTParameter(gd, forVirtualBase, delegating);
}

// The idea here is creating a separate block for the throw with an
// `UnreachableOp` as the terminator. So, we branch from the current block
// to the throw block and create a block for the remaining operations.
static void insertThrowAndSplit(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value exceptionPtr = {},
                                mlir::FlatSymbolRefAttr typeInfo = {},
                                mlir::FlatSymbolRefAttr dtor = {}) {
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Region *region = currentBlock->getParent();

  if (currentBlock->empty()) {
    cir::ThrowOp::create(builder, loc, exceptionPtr, typeInfo, dtor);
    cir::UnreachableOp::create(builder, loc);
  } else {
    mlir::Block *throwBlock = builder.createBlock(region);

    cir::ThrowOp::create(builder, loc, exceptionPtr, typeInfo, dtor);
    cir::UnreachableOp::create(builder, loc);

    builder.setInsertionPointToEnd(currentBlock);
    cir::BrOp::create(builder, loc, throwBlock);
  }

  (void)builder.createBlock(region);
}

void CIRGenItaniumCXXABI::emitRethrow(CIRGenFunction &cgf, bool isNoReturn) {
  // void __cxa_rethrow();
  if (isNoReturn) {
    CIRGenBuilderTy &builder = cgf.getBuilder();
    assert(cgf.currSrcLoc && "expected source location");
    mlir::Location loc = *cgf.currSrcLoc;
    insertThrowAndSplit(builder, loc);
  } else {
    cgm.errorNYI("emitRethrow with isNoReturn false");
  }
}

void CIRGenItaniumCXXABI::emitThrow(CIRGenFunction &cgf,
                                    const CXXThrowExpr *e) {
  // This differs a bit from LLVM codegen, CIR has native operations for some
  // cxa functions, and defers allocation size computation, always pass the dtor
  // symbol, etc. CIRGen also does not use getAllocateExceptionFn / getThrowFn.

  // Now allocate the exception object.
  CIRGenBuilderTy &builder = cgf.getBuilder();
  QualType clangThrowType = e->getSubExpr()->getType();
  cir::PointerType throwTy =
      builder.getPointerTo(cgf.convertType(clangThrowType));
  uint64_t typeSize =
      cgf.getContext().getTypeSizeInChars(clangThrowType).getQuantity();
  mlir::Location subExprLoc = cgf.getLoc(e->getSubExpr()->getSourceRange());

  // Defer computing allocation size to some later lowering pass.
  mlir::TypedValue<cir::PointerType> exceptionPtr =
      cir::AllocExceptionOp::create(builder, subExprLoc, throwTy,
                                    builder.getI64IntegerAttr(typeSize))
          .getAddr();

  // Build expression and store its result into exceptionPtr.
  CharUnits exnAlign = cgf.getContext().getExnObjectAlignment();
  cgf.emitAnyExprToExn(e->getSubExpr(), Address(exceptionPtr, exnAlign));

  // Get the RTTI symbol address.
  auto typeInfo = mlir::cast<cir::GlobalViewAttr>(
      cgm.getAddrOfRTTIDescriptor(subExprLoc, clangThrowType,
                                  /*forEH=*/true));
  assert(!typeInfo.getIndices() && "expected no indirection");

  // The address of the destructor.
  //
  // Note: LLVM codegen already optimizes out the dtor if the
  // type is a record with trivial dtor (by passing down a
  // null dtor). In CIR, we forward this info and allow for
  // Lowering pass to skip passing the trivial function.
  //
  if (const RecordType *recordTy = clangThrowType->getAs<RecordType>()) {
    CXXRecordDecl *rec =
        cast<CXXRecordDecl>(recordTy->getOriginalDecl()->getDefinition());
    assert(!cir::MissingFeatures::isTrivialCtorOrDtor());
    if (!rec->hasTrivialDestructor()) {
      cgm.errorNYI("emitThrow: non-trivial destructor");
      return;
    }
  }

  // Now throw the exception.
  mlir::Location loc = cgf.getLoc(e->getSourceRange());
  insertThrowAndSplit(builder, loc, exceptionPtr, typeInfo.getSymbol());
}

CIRGenCXXABI *clang::CIRGen::CreateCIRGenItaniumCXXABI(CIRGenModule &cgm) {
  switch (cgm.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
    return new CIRGenItaniumCXXABI(cgm);

  case TargetCXXABI::AppleARM64:
    // The general Itanium ABI will do until we implement something that
    // requires special handling.
    assert(!cir::MissingFeatures::cxxabiAppleARM64CXXABI());
    return new CIRGenItaniumCXXABI(cgm);

  default:
    llvm_unreachable("bad or NYI ABI kind");
  }
}

cir::GlobalOp CIRGenItaniumCXXABI::getAddrOfVTable(const CXXRecordDecl *rd,
                                                   CharUnits vptrOffset) {
  assert(vptrOffset.isZero() && "Itanium ABI only supports zero vptr offsets");
  cir::GlobalOp &vtable = vtables[rd];
  if (vtable)
    return vtable;

  // Queue up this vtable for possible deferred emission.
  assert(!cir::MissingFeatures::deferredVtables());

  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  getMangleContext().mangleCXXVTable(rd, out);

  const VTableLayout &vtLayout =
      cgm.getItaniumVTableContext().getVTableLayout(rd);
  mlir::Type vtableType = cgm.getVTables().getVTableType(vtLayout);

  // Use pointer alignment for the vtable. Otherwise we would align them based
  // on the size of the initializer which doesn't make sense as only single
  // values are read.
  unsigned ptrAlign = cgm.getItaniumVTableContext().isRelativeLayout()
                          ? 32
                          : cgm.getTarget().getPointerAlign(LangAS::Default);

  vtable = cgm.createOrReplaceCXXRuntimeVariable(
      cgm.getLoc(rd->getSourceRange()), name, vtableType,
      cir::GlobalLinkageKind::ExternalLinkage,
      cgm.getASTContext().toCharUnitsFromBits(ptrAlign));
  // LLVM codegen handles unnamedAddr
  assert(!cir::MissingFeatures::opGlobalUnnamedAddr());

  // In MS C++ if you have a class with virtual functions in which you are using
  // selective member import/export, then all virtual functions must be exported
  // unless they are inline, otherwise a link error will result. To match this
  // behavior, for such classes, we dllimport the vtable if it is defined
  // externally and all the non-inline virtual methods are marked dllimport, and
  // we dllexport the vtable if it is defined in this TU and all the non-inline
  // virtual methods are marked dllexport.
  if (cgm.getTarget().hasPS4DLLImportExport())
    cgm.errorNYI(rd->getSourceRange(),
                 "getAddrOfVTable: PS4 DLL import/export");

  cgm.setGVProperties(vtable, rd);
  return vtable;
}

CIRGenCallee CIRGenItaniumCXXABI::getVirtualFunctionPointer(
    CIRGenFunction &cgf, clang::GlobalDecl gd, Address thisAddr, mlir::Type ty,
    SourceLocation srcLoc) {
  CIRGenBuilderTy &builder = cgm.getBuilder();
  mlir::Location loc = cgf.getLoc(srcLoc);
  cir::PointerType tyPtr = builder.getPointerTo(ty);
  auto *methodDecl = cast<CXXMethodDecl>(gd.getDecl());
  mlir::Value vtable = cgf.getVTablePtr(loc, thisAddr, methodDecl->getParent());

  uint64_t vtableIndex = cgm.getItaniumVTableContext().getMethodVTableIndex(gd);
  mlir::Value vfunc{};
  if (cgf.shouldEmitVTableTypeCheckedLoad(methodDecl->getParent())) {
    cgm.errorNYI(loc, "getVirtualFunctionPointer: emitVTableTypeCheckedLoad");
  } else {
    assert(!cir::MissingFeatures::emitTypeMetadataCodeForVCall());

    mlir::Value vfuncLoad;
    if (cgm.getItaniumVTableContext().isRelativeLayout()) {
      assert(!cir::MissingFeatures::vtableRelativeLayout());
      cgm.errorNYI(loc, "getVirtualFunctionPointer: isRelativeLayout");
    } else {
      auto vtableSlotPtr = cir::VTableGetVirtualFnAddrOp::create(
          builder, loc, builder.getPointerTo(tyPtr), vtable, vtableIndex);
      vfuncLoad = builder.createAlignedLoad(loc, tyPtr, vtableSlotPtr,
                                            cgf.getPointerAlign());
    }

    // Add !invariant.load md to virtual function load to indicate that
    // function didn't change inside vtable.
    // It's safe to add it without -fstrict-vtable-pointers, but it would not
    // help in devirtualization because it will only matter if we will have 2
    // the same virtual function loads from the same vtable load, which won't
    // happen without enabled devirtualization with -fstrict-vtable-pointers.
    if (cgm.getCodeGenOpts().OptimizationLevel > 0 &&
        cgm.getCodeGenOpts().StrictVTablePointers) {
      cgm.errorNYI(loc, "getVirtualFunctionPointer: strictVTablePointers");
    }
    vfunc = vfuncLoad;
  }

  CIRGenCallee callee(gd, vfunc.getDefiningOp());
  return callee;
}

mlir::Value CIRGenItaniumCXXABI::getVTableAddressPointInStructorWithVTT(
    CIRGenFunction &cgf, const CXXRecordDecl *vtableClass, BaseSubobject base,
    const CXXRecordDecl *nearestVBase) {
  assert((base.getBase()->getNumVBases() || nearestVBase != nullptr) &&
         needsVTTParameter(cgf.curGD) && "This class doesn't have VTT");

  // Get the secondary vpointer index.
  uint64_t virtualPointerIndex =
      cgm.getVTables().getSecondaryVirtualPointerIndex(vtableClass, base);

  /// Load the VTT.
  mlir::Value vttPtr = cgf.loadCXXVTT();
  mlir::Location loc = cgf.getLoc(vtableClass->getSourceRange());
  // Calculate the address point from the VTT, and the offset may be zero.
  vttPtr = cgf.getBuilder().createVTTAddrPoint(loc, vttPtr.getType(), vttPtr,
                                               virtualPointerIndex);
  // And load the address point from the VTT.
  auto vptrType = cir::VPtrType::get(cgf.getBuilder().getContext());
  return cgf.getBuilder().createAlignedLoad(loc, vptrType, vttPtr,
                                            cgf.getPointerAlign());
}

mlir::Value
CIRGenItaniumCXXABI::getVTableAddressPoint(BaseSubobject base,
                                           const CXXRecordDecl *vtableClass) {
  cir::GlobalOp vtable = getAddrOfVTable(vtableClass, CharUnits());

  // Find the appropriate vtable within the vtable group, and the address point
  // within that vtable.
  VTableLayout::AddressPointLocation addressPoint =
      cgm.getItaniumVTableContext()
          .getVTableLayout(vtableClass)
          .getAddressPoint(base);

  mlir::OpBuilder &builder = cgm.getBuilder();
  auto vtablePtrTy = cir::VPtrType::get(builder.getContext());

  return builder.create<cir::VTableAddrPointOp>(
      cgm.getLoc(vtableClass->getSourceRange()), vtablePtrTy,
      mlir::FlatSymbolRefAttr::get(vtable.getSymNameAttr()),
      cir::AddressPointAttr::get(cgm.getBuilder().getContext(),
                                 addressPoint.VTableIndex,
                                 addressPoint.AddressPointIndex));
}

mlir::Value CIRGenItaniumCXXABI::getVTableAddressPointInStructor(
    CIRGenFunction &cgf, const clang::CXXRecordDecl *vtableClass,
    clang::BaseSubobject base, const clang::CXXRecordDecl *nearestVBase) {

  if ((base.getBase()->getNumVBases() || nearestVBase != nullptr) &&
      needsVTTParameter(cgf.curGD)) {
    return getVTableAddressPointInStructorWithVTT(cgf, vtableClass, base,
                                                  nearestVBase);
  }
  return getVTableAddressPoint(base, vtableClass);
}

bool CIRGenItaniumCXXABI::isVirtualOffsetNeededForVTableField(
    CIRGenFunction &cgf, CIRGenFunction::VPtr vptr) {
  if (vptr.nearestVBase == nullptr)
    return false;
  return needsVTTParameter(cgf.curGD);
}

mlir::Value CIRGenItaniumCXXABI::getVirtualBaseClassOffset(
    mlir::Location loc, CIRGenFunction &cgf, Address thisAddr,
    const CXXRecordDecl *classDecl, const CXXRecordDecl *baseClassDecl) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Value vtablePtr = cgf.getVTablePtr(loc, thisAddr, classDecl);
  mlir::Value vtableBytePtr = builder.createBitcast(vtablePtr, cgm.UInt8PtrTy);
  CharUnits vbaseOffsetOffset =
      cgm.getItaniumVTableContext().getVirtualBaseOffsetOffset(classDecl,
                                                               baseClassDecl);
  mlir::Value offsetVal =
      builder.getSInt64(vbaseOffsetOffset.getQuantity(), loc);
  auto vbaseOffsetPtr = cir::PtrStrideOp::create(builder, loc, cgm.UInt8PtrTy,
                                                 vtableBytePtr, offsetVal);

  mlir::Value vbaseOffset;
  if (cgm.getItaniumVTableContext().isRelativeLayout()) {
    assert(!cir::MissingFeatures::vtableRelativeLayout());
    cgm.errorNYI(loc, "getVirtualBaseClassOffset: relative layout");
  } else {
    mlir::Value offsetPtr = builder.createBitcast(
        vbaseOffsetPtr, builder.getPointerTo(cgm.PtrDiffTy));
    vbaseOffset = builder.createLoad(
        loc, Address(offsetPtr, cgm.PtrDiffTy, cgf.getPointerAlign()));
  }
  return vbaseOffset;
}

static cir::FuncOp getBadCastFn(CIRGenFunction &cgf) {
  // Prototype: void __cxa_bad_cast();

  // TODO(cir): set the calling convention of the runtime function.
  assert(!cir::MissingFeatures::opFuncCallingConv());

  cir::FuncType fnTy =
      cgf.getBuilder().getFuncType({}, cgf.getBuilder().getVoidTy());
  return cgf.cgm.createRuntimeFunction(fnTy, "__cxa_bad_cast");
}

// TODO(cir): This could be shared with classic codegen.
static CharUnits computeOffsetHint(ASTContext &astContext,
                                   const CXXRecordDecl *src,
                                   const CXXRecordDecl *dst) {
  CXXBasePaths paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);

  // If Dst is not derived from Src we can skip the whole computation below and
  // return that Src is not a public base of Dst.  Record all inheritance paths.
  if (!dst->isDerivedFrom(src, paths))
    return CharUnits::fromQuantity(-2ULL);

  unsigned numPublicPaths = 0;
  CharUnits offset;

  // Now walk all possible inheritance paths.
  for (const CXXBasePath &path : paths) {
    if (path.Access != AS_public) // Ignore non-public inheritance.
      continue;

    ++numPublicPaths;

    for (const CXXBasePathElement &pathElement : path) {
      // If the path contains a virtual base class we can't give any hint.
      // -1: no hint.
      if (pathElement.Base->isVirtual())
        return CharUnits::fromQuantity(-1ULL);

      if (numPublicPaths > 1) // Won't use offsets, skip computation.
        continue;

      // Accumulate the base class offsets.
      const ASTRecordLayout &L =
          astContext.getASTRecordLayout(pathElement.Class);
      offset += L.getBaseClassOffset(
          pathElement.Base->getType()->getAsCXXRecordDecl());
    }
  }

  // -2: Src is not a public base of Dst.
  if (numPublicPaths == 0)
    return CharUnits::fromQuantity(-2ULL);

  // -3: Src is a multiple public base type but never a virtual base type.
  if (numPublicPaths > 1)
    return CharUnits::fromQuantity(-3ULL);

  // Otherwise, the Src type is a unique public nonvirtual base type of Dst.
  // Return the offset of Src from the origin of Dst.
  return offset;
}

static cir::FuncOp getItaniumDynamicCastFn(CIRGenFunction &cgf) {
  // Prototype:
  // void *__dynamic_cast(const void *sub,
  //                      global_as const abi::__class_type_info *src,
  //                      global_as const abi::__class_type_info *dst,
  //                      std::ptrdiff_t src2dst_offset);

  mlir::Type voidPtrTy = cgf.getBuilder().getVoidPtrTy();
  mlir::Type rttiPtrTy = cgf.getBuilder().getUInt8PtrTy();
  mlir::Type ptrDiffTy = cgf.convertType(cgf.getContext().getPointerDiffType());

  // TODO(cir): mark the function as nowind willreturn readonly.
  assert(!cir::MissingFeatures::opFuncNoUnwind());
  assert(!cir::MissingFeatures::opFuncWillReturn());
  assert(!cir::MissingFeatures::opFuncReadOnly());

  // TODO(cir): set the calling convention of the runtime function.
  assert(!cir::MissingFeatures::opFuncCallingConv());

  cir::FuncType FTy = cgf.getBuilder().getFuncType(
      {voidPtrTy, rttiPtrTy, rttiPtrTy, ptrDiffTy}, voidPtrTy);
  return cgf.cgm.createRuntimeFunction(FTy, "__dynamic_cast");
}

static cir::DynamicCastInfoAttr emitDynamicCastInfo(CIRGenFunction &cgf,
                                                    mlir::Location loc,
                                                    QualType srcRecordTy,
                                                    QualType destRecordTy) {
  auto srcRtti = mlir::cast<cir::GlobalViewAttr>(
      cgf.cgm.getAddrOfRTTIDescriptor(loc, srcRecordTy));
  auto destRtti = mlir::cast<cir::GlobalViewAttr>(
      cgf.cgm.getAddrOfRTTIDescriptor(loc, destRecordTy));

  cir::FuncOp runtimeFuncOp = getItaniumDynamicCastFn(cgf);
  cir::FuncOp badCastFuncOp = getBadCastFn(cgf);
  auto runtimeFuncRef = mlir::FlatSymbolRefAttr::get(runtimeFuncOp);
  auto badCastFuncRef = mlir::FlatSymbolRefAttr::get(badCastFuncOp);

  const CXXRecordDecl *srcDecl = srcRecordTy->getAsCXXRecordDecl();
  const CXXRecordDecl *destDecl = destRecordTy->getAsCXXRecordDecl();
  CharUnits offsetHint = computeOffsetHint(cgf.getContext(), srcDecl, destDecl);

  mlir::Type ptrdiffTy = cgf.convertType(cgf.getContext().getPointerDiffType());
  auto offsetHintAttr = cir::IntAttr::get(ptrdiffTy, offsetHint.getQuantity());

  return cir::DynamicCastInfoAttr::get(srcRtti, destRtti, runtimeFuncRef,
                                       badCastFuncRef, offsetHintAttr);
}

mlir::Value CIRGenItaniumCXXABI::emitDynamicCast(CIRGenFunction &cgf,
                                                 mlir::Location loc,
                                                 QualType srcRecordTy,
                                                 QualType destRecordTy,
                                                 cir::PointerType destCIRTy,
                                                 bool isRefCast, Address src) {
  bool isCastToVoid = destRecordTy.isNull();
  assert((!isCastToVoid || !isRefCast) && "cannot cast to void reference");

  if (isCastToVoid) {
    cgm.errorNYI(loc, "emitDynamicCastToVoid");
    return {};
  }

  // If the destination is effectively final, the cast succeeds if and only
  // if the dynamic type of the pointer is exactly the destination type.
  if (destRecordTy->getAsCXXRecordDecl()->isEffectivelyFinal() &&
      cgf.cgm.getCodeGenOpts().OptimizationLevel > 0) {
    cgm.errorNYI(loc, "emitExactDynamicCast");
    return {};
  }

  cir::DynamicCastInfoAttr castInfo =
      emitDynamicCastInfo(cgf, loc, srcRecordTy, destRecordTy);
  return cgf.getBuilder().createDynCast(loc, src.getPointer(), destCIRTy,
                                        isRefCast, castInfo);
}
