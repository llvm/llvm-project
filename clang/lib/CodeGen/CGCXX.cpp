//===--- CGCXX.cpp - Emit LLVM Code for declarations ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy

#include "CodeGenModule.h"
#include "CGCXXABI.h"
#include "CodeGenFunction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

/// Try to emit a base destructor as an alias to its primary
/// base-class destructor.
bool CodeGenModule::TryEmitBaseDestructorAsAlias(const CXXDestructorDecl *D) {
  if (!getCodeGenOpts().CXXCtorDtorAliases)
    return true;

  // Producing an alias to a base class ctor/dtor can degrade debug quality
  // as the debugger cannot tell them appart.
  if (getCodeGenOpts().OptimizationLevel == 0)
    return true;

  // If the destructor doesn't have a trivial body, we have to emit it
  // separately.
  if (!D->hasTrivialBody())
    return true;

  const CXXRecordDecl *Class = D->getParent();

  // If we need to manipulate a VTT parameter, give up.
  if (Class->getNumVBases()) {
    // Extra Credit:  passing extra parameters is perfectly safe
    // in many calling conventions, so only bail out if the ctor's
    // calling convention is nonstandard.
    return true;
  }

  // If any field has a non-trivial destructor, we have to emit the
  // destructor separately.
  for (CXXRecordDecl::field_iterator I = Class->field_begin(),
         E = Class->field_end(); I != E; ++I)
    if (I->getType().isDestructedType())
      return true;

  // Try to find a unique base class with a non-trivial destructor.
  const CXXRecordDecl *UniqueBase = 0;
  for (CXXRecordDecl::base_class_const_iterator I = Class->bases_begin(),
         E = Class->bases_end(); I != E; ++I) {

    // We're in the base destructor, so skip virtual bases.
    if (I->isVirtual()) continue;

    // Skip base classes with trivial destructors.
    const CXXRecordDecl *Base
      = cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    if (Base->hasTrivialDestructor()) continue;

    // If we've already found a base class with a non-trivial
    // destructor, give up.
    if (UniqueBase) return true;
    UniqueBase = Base;
  }

  // If we didn't find any bases with a non-trivial destructor, then
  // the base destructor is actually effectively trivial, which can
  // happen if it was needlessly user-defined or if there are virtual
  // bases with non-trivial destructors.
  if (!UniqueBase)
    return true;

  // If the base is at a non-zero offset, give up.
  const ASTRecordLayout &ClassLayout = Context.getASTRecordLayout(Class);
  if (!ClassLayout.getBaseClassOffset(UniqueBase).isZero())
    return true;

  const CXXDestructorDecl *BaseD = UniqueBase->getDestructor();
  return TryEmitDefinitionAsAlias(GlobalDecl(D, Dtor_Base),
                                  GlobalDecl(BaseD, Dtor_Base),
                                  false);
}

/// Try to emit a definition as a global alias for another definition.
/// If \p InEveryTU is true, we know that an equivalent alias can be produced
/// in every translation unit.
bool CodeGenModule::TryEmitDefinitionAsAlias(GlobalDecl AliasDecl,
                                             GlobalDecl TargetDecl,
                                             bool InEveryTU) {
  if (!getCodeGenOpts().CXXCtorDtorAliases)
    return true;

  // The alias will use the linkage of the referrent.  If we can't
  // support aliases with that linkage, fail.
  llvm::GlobalValue::LinkageTypes Linkage = getFunctionLinkage(AliasDecl);

  // We can't use an alias if the linkage is not valid for one.
  if (!llvm::GlobalAlias::isValidLinkage(Linkage))
    return true;

  llvm::GlobalValue::LinkageTypes TargetLinkage =
      getFunctionLinkage(TargetDecl);

  // Check if we have it already.
  StringRef MangledName = getMangledName(AliasDecl);
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry && !Entry->isDeclaration())
    return false;
  if (Replacements.count(MangledName))
    return false;

  // Derive the type for the alias.
  llvm::PointerType *AliasType
    = getTypes().GetFunctionType(AliasDecl)->getPointerTo();

  // Find the referrent.  Some aliases might require a bitcast, in
  // which case the caller is responsible for ensuring the soundness
  // of these semantics.
  llvm::GlobalValue *Ref = cast<llvm::GlobalValue>(GetAddrOfGlobal(TargetDecl));
  llvm::Constant *Aliasee = Ref;
  if (Ref->getType() != AliasType)
    Aliasee = llvm::ConstantExpr::getBitCast(Ref, AliasType);

  // Instead of creating as alias to a linkonce_odr, replace all of the uses
  // of the aliassee.
  if (llvm::GlobalValue::isDiscardableIfUnused(Linkage) &&
     (TargetLinkage != llvm::GlobalValue::AvailableExternallyLinkage ||
      !TargetDecl.getDecl()->hasAttr<AlwaysInlineAttr>())) {
    // FIXME: An extern template instanciation will create functions with
    // linkage "AvailableExternally". In libc++, some classes also define
    // members with attribute "AlwaysInline" and expect no reference to
    // be generated. It is desirable to reenable this optimisation after
    // corresponding LLVM changes.
    Replacements[MangledName] = Aliasee;
    return false;
  }

  if (!InEveryTU) {
    /// If we don't have a definition for the destructor yet, don't
    /// emit.  We can't emit aliases to declarations; that's just not
    /// how aliases work.
    if (Ref->isDeclaration())
      return true;
  }

  // Don't create an alias to a linker weak symbol. This avoids producing
  // different COMDATs in different TUs. Another option would be to
  // output the alias both for weak_odr and linkonce_odr, but that
  // requires explicit comdat support in the IL.
  if (llvm::GlobalValue::isWeakForLinker(TargetLinkage))
    return true;

  // Create the alias with no name.
  llvm::GlobalAlias *Alias = 
    new llvm::GlobalAlias(AliasType, Linkage, "", Aliasee, &getModule());

  // Switch any previous uses to the alias.
  if (Entry) {
    assert(Entry->getType() == AliasType &&
           "declaration exists with different type");
    Alias->takeName(Entry);
    Entry->replaceAllUsesWith(Alias);
    Entry->eraseFromParent();
  } else {
    Alias->setName(MangledName);
  }

  // Finally, set up the alias with its proper name and attributes.
  SetCommonAttributes(cast<NamedDecl>(AliasDecl.getDecl()), Alias);

  return false;
}

void CodeGenModule::EmitCXXConstructor(const CXXConstructorDecl *ctor,
                                       CXXCtorType ctorType) {
  // The complete constructor is equivalent to the base constructor
  // for classes with no virtual bases.  Try to emit it as an alias.
  if (getTarget().getCXXABI().hasConstructorVariants() &&
      !ctor->getParent()->getNumVBases() &&
      (ctorType == Ctor_Complete || ctorType == Ctor_Base)) {
    bool ProducedAlias =
        !TryEmitDefinitionAsAlias(GlobalDecl(ctor, Ctor_Complete),
                                  GlobalDecl(ctor, Ctor_Base), true);
    if (ctorType == Ctor_Complete && ProducedAlias)
      return;
  }

  const CGFunctionInfo &fnInfo =
    getTypes().arrangeCXXConstructorDeclaration(ctor, ctorType);

  llvm::Function *fn =
    cast<llvm::Function>(GetAddrOfCXXConstructor(ctor, ctorType, &fnInfo));
  setFunctionLinkage(GlobalDecl(ctor, ctorType), fn);

  CodeGenFunction(*this).GenerateCode(GlobalDecl(ctor, ctorType), fn, fnInfo);

  SetFunctionDefinitionAttributes(ctor, fn);
  SetLLVMFunctionAttributesForDefinition(ctor, fn);
}

llvm::GlobalValue *
CodeGenModule::GetAddrOfCXXConstructor(const CXXConstructorDecl *ctor,
                                       CXXCtorType ctorType,
                                       const CGFunctionInfo *fnInfo) {
  GlobalDecl GD(ctor, ctorType);
  
  StringRef name = getMangledName(GD);
  if (llvm::GlobalValue *existing = GetGlobalValue(name))
    return existing;

  if (!fnInfo)
    fnInfo = &getTypes().arrangeCXXConstructorDeclaration(ctor, ctorType);

  llvm::FunctionType *fnType = getTypes().GetFunctionType(*fnInfo);
  return cast<llvm::Function>(GetOrCreateLLVMFunction(name, fnType, GD,
                                                      /*ForVTable=*/false));
}

void CodeGenModule::EmitCXXDestructor(const CXXDestructorDecl *dtor,
                                      CXXDtorType dtorType) {
  // The complete destructor is equivalent to the base destructor for
  // classes with no virtual bases, so try to emit it as an alias.
  if (!dtor->getParent()->getNumVBases() &&
      (dtorType == Dtor_Complete || dtorType == Dtor_Base)) {
    bool ProducedAlias =
        !TryEmitDefinitionAsAlias(GlobalDecl(dtor, Dtor_Complete),
                                  GlobalDecl(dtor, Dtor_Base), true);
    if (ProducedAlias) {
      if (dtorType == Dtor_Complete)
        return;
      if (dtor->isVirtual())
        getVTables().EmitThunks(GlobalDecl(dtor, Dtor_Complete));
    }
  }

  // The base destructor is equivalent to the base destructor of its
  // base class if there is exactly one non-virtual base class with a
  // non-trivial destructor, there are no fields with a non-trivial
  // destructor, and the body of the destructor is trivial.
  if (dtorType == Dtor_Base && !TryEmitBaseDestructorAsAlias(dtor))
    return;

  const CGFunctionInfo &fnInfo =
    getTypes().arrangeCXXDestructor(dtor, dtorType);

  llvm::Function *fn =
    cast<llvm::Function>(GetAddrOfCXXDestructor(dtor, dtorType, &fnInfo));
  setFunctionLinkage(GlobalDecl(dtor, dtorType), fn);

  CodeGenFunction(*this).GenerateCode(GlobalDecl(dtor, dtorType), fn, fnInfo);

  SetFunctionDefinitionAttributes(dtor, fn);
  SetLLVMFunctionAttributesForDefinition(dtor, fn);
}

llvm::GlobalValue *
CodeGenModule::GetAddrOfCXXDestructor(const CXXDestructorDecl *dtor,
                                      CXXDtorType dtorType,
                                      const CGFunctionInfo *fnInfo,
                                      llvm::FunctionType *fnType) {
  GlobalDecl GD(dtor, dtorType);

  StringRef name = getMangledName(GD);
  if (llvm::GlobalValue *existing = GetGlobalValue(name))
    return existing;

  if (!fnType) {
    if (!fnInfo) fnInfo = &getTypes().arrangeCXXDestructor(dtor, dtorType);
    fnType = getTypes().GetFunctionType(*fnInfo);
  }
  return cast<llvm::Function>(GetOrCreateLLVMFunction(name, fnType, GD,
                                                      /*ForVTable=*/false));
}

static llvm::Value *BuildAppleKextVirtualCall(CodeGenFunction &CGF,
                                              GlobalDecl GD,
                                              llvm::Type *Ty,
                                              const CXXRecordDecl *RD) {
  assert(!CGF.CGM.getTarget().getCXXABI().isMicrosoft() &&
         "No kext in Microsoft ABI");
  GD = GD.getCanonicalDecl();
  CodeGenModule &CGM = CGF.CGM;
  llvm::Value *VTable = CGM.getCXXABI().getAddrOfVTable(RD, CharUnits());
  Ty = Ty->getPointerTo()->getPointerTo();
  VTable = CGF.Builder.CreateBitCast(VTable, Ty);
  assert(VTable && "BuildVirtualCall = kext vtbl pointer is null");
  uint64_t VTableIndex = CGM.getItaniumVTableContext().getMethodVTableIndex(GD);
  uint64_t AddressPoint =
    CGM.getItaniumVTableContext().getVTableLayout(RD)
       .getAddressPoint(BaseSubobject(RD, CharUnits::Zero()));
  VTableIndex += AddressPoint;
  llvm::Value *VFuncPtr =
    CGF.Builder.CreateConstInBoundsGEP1_64(VTable, VTableIndex, "vfnkxt");
  return CGF.Builder.CreateLoad(VFuncPtr);
}

/// BuildAppleKextVirtualCall - This routine is to support gcc's kext ABI making
/// indirect call to virtual functions. It makes the call through indexing
/// into the vtable.
llvm::Value *
CodeGenFunction::BuildAppleKextVirtualCall(const CXXMethodDecl *MD, 
                                  NestedNameSpecifier *Qual,
                                  llvm::Type *Ty) {
  assert((Qual->getKind() == NestedNameSpecifier::TypeSpec) &&
         "BuildAppleKextVirtualCall - bad Qual kind");
  
  const Type *QTy = Qual->getAsType();
  QualType T = QualType(QTy, 0);
  const RecordType *RT = T->getAs<RecordType>();
  assert(RT && "BuildAppleKextVirtualCall - Qual type must be record");
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
  
  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD))
    return BuildAppleKextVirtualDestructorCall(DD, Dtor_Complete, RD);

  return ::BuildAppleKextVirtualCall(*this, MD, Ty, RD);
}

/// BuildVirtualCall - This routine makes indirect vtable call for
/// call to virtual destructors. It returns 0 if it could not do it.
llvm::Value *
CodeGenFunction::BuildAppleKextVirtualDestructorCall(
                                            const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            const CXXRecordDecl *RD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(DD);
  // FIXME. Dtor_Base dtor is always direct!!
  // It need be somehow inline expanded into the caller.
  // -O does that. But need to support -O0 as well.
  if (MD->isVirtual() && Type != Dtor_Base) {
    // Compute the function type we're calling.
    const CGFunctionInfo &FInfo =
      CGM.getTypes().arrangeCXXDestructor(DD, Dtor_Complete);
    llvm::Type *Ty = CGM.getTypes().GetFunctionType(FInfo);
    return ::BuildAppleKextVirtualCall(*this, GlobalDecl(DD, Type), Ty, RD);
  }
  return 0;
}
