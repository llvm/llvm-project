//===--- CGCXX.cpp - Emit LLVM Code for declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang;
using namespace cir;

/// Try to emit a base destructor as an alias to its primary
/// base-class destructor.
bool CIRGenModule::tryEmitBaseDestructorAsAlias(const CXXDestructorDecl *D) {
  if (!getCodeGenOpts().CXXCtorDtorAliases)
    return true;

  // Producing an alias to a base class ctor/dtor can degrade debug quality
  // as the debugger cannot tell them apart.
  if (getCodeGenOpts().OptimizationLevel == 0)
    return true;

  // If sanitizing memory to check for use-after-dtor, do not emit as
  //  an alias, unless this class owns no members.
  if (getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
      !D->getParent()->field_empty())
    assert(!UnimplementedFeature::sanitizeDtor());

  // If the destructor doesn't have a trivial body, we have to emit it
  // separately.
  if (!D->hasTrivialBody())
    return true;

  const CXXRecordDecl *Class = D->getParent();

  // We are going to instrument this destructor, so give up even if it is
  // currently empty.
  if (Class->mayInsertExtraPadding())
    return true;

  // If we need to manipulate a VTT parameter, give up.
  if (Class->getNumVBases()) {
    // Extra Credit:  passing extra parameters is perfectly safe
    // in many calling conventions, so only bail out if the ctor's
    // calling convention is nonstandard.
    return true;
  }

  // If any field has a non-trivial destructor, we have to emit the
  // destructor separately.
  for (const auto *I : Class->fields())
    if (I->getType().isDestructedType())
      return true;

  // Try to find a unique base class with a non-trivial destructor.
  const CXXRecordDecl *UniqueBase = nullptr;
  for (const auto &I : Class->bases()) {

    // We're in the base destructor, so skip virtual bases.
    if (I.isVirtual())
      continue;

    // Skip base classes with trivial destructors.
    const auto *Base =
        cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());
    if (Base->hasTrivialDestructor())
      continue;

    // If we've already found a base class with a non-trivial
    // destructor, give up.
    if (UniqueBase)
      return true;
    UniqueBase = Base;
  }

  // If we didn't find any bases with a non-trivial destructor, then
  // the base destructor is actually effectively trivial, which can
  // happen if it was needlessly user-defined or if there are virtual
  // bases with non-trivial destructors.
  if (!UniqueBase)
    return true;

  // If the base is at a non-zero offset, give up.
  const ASTRecordLayout &ClassLayout = astCtx.getASTRecordLayout(Class);
  if (!ClassLayout.getBaseClassOffset(UniqueBase).isZero())
    return true;

  // Give up if the calling conventions don't match. We could update the call,
  // but it is probably not worth it.
  const CXXDestructorDecl *BaseD = UniqueBase->getDestructor();
  if (BaseD->getType()->castAs<FunctionType>()->getCallConv() !=
      D->getType()->castAs<FunctionType>()->getCallConv())
    return true;

  GlobalDecl AliasDecl(D, Dtor_Base);
  GlobalDecl TargetDecl(BaseD, Dtor_Base);

  // The alias will use the linkage of the referent.  If we can't
  // support aliases with that linkage, fail.
  auto Linkage = getFunctionLinkage(AliasDecl);

  // We can't use an alias if the linkage is not valid for one.
  if (!mlir::cir::isValidLinkage(Linkage))
    return true;

  auto TargetLinkage = getFunctionLinkage(TargetDecl);

  // Check if we have it already.
  StringRef MangledName = getMangledName(AliasDecl);
  auto Entry = getGlobalValue(MangledName);
  auto fnOp = dyn_cast_or_null<mlir::cir::FuncOp>(Entry);
  if (Entry && fnOp && !fnOp.isDeclaration())
    return false;
  if (Replacements.count(MangledName))
    return false;

  assert(fnOp && "only knows how to handle FuncOp");
  [[maybe_unused]] auto AliasValueType = getTypes().GetFunctionType(AliasDecl);

  // Find the referent.
  auto Aliasee = cast<mlir::cir::FuncOp>(GetAddrOfGlobal(TargetDecl));

  // Instead of creating as alias to a linkonce_odr, replace all of the uses
  // of the aliasee.
  if (mlir::cir::isDiscardableIfUnused(Linkage) &&
      !(TargetLinkage ==
            mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage &&
        TargetDecl.getDecl()->hasAttr<AlwaysInlineAttr>())) {
    // FIXME: An extern template instantiation will create functions with
    // linkage "AvailableExternally". In libc++, some classes also define
    // members with attribute "AlwaysInline" and expect no reference to
    // be generated. It is desirable to reenable this optimisation after
    // corresponding LLVM changes.
    llvm_unreachable("NYI");
  }

  // If we have a weak, non-discardable alias (weak, weak_odr), like an
  // extern template instantiation or a dllexported class, avoid forming it on
  // COFF. A COFF weak external alias cannot satisfy a normal undefined
  // symbol reference from another TU. The other TU must also mark the
  // referenced symbol as weak, which we cannot rely on.
  if (mlir::cir::isWeakForLinker(Linkage) && getTriple().isOSBinFormatCOFF()) {
    llvm_unreachable("NYI");
  }

  // If we don't have a definition for the destructor yet or the definition
  // is
  // avaialable_externally, don't emit an alias.  We can't emit aliases to
  // declarations; that's just not how aliases work.
  if (Aliasee.isDeclarationForLinker())
    return true;

  // Don't create an alias to a linker weak symbol. This avoids producing
  // different COMDATs in different TUs. Another option would be to
  // output the alias both for weak_odr and linkonce_odr, but that
  // requires explicit comdat support in the IL.
  if (mlir::cir::isWeakForLinker(TargetLinkage))
    llvm_unreachable("NYI");

  // Create the alias with no name.
  auto *AliasFD = dyn_cast<FunctionDecl>(AliasDecl.getDecl());
  assert(AliasFD && "expected FunctionDecl");
  auto Alias = createCIRFunction(getLoc(AliasDecl.getDecl()->getSourceRange()),
                                 "", Aliasee.getFunctionType(), AliasFD);
  Alias.setAliasee(Aliasee.getName());
  Alias.setLinkage(Linkage);
  mlir::SymbolTable::setSymbolVisibility(
      Alias, getMLIRVisibilityFromCIRLinkage(Linkage));

  // Alias constructors and destructors are always unnamed_addr.
  assert(!UnimplementedFeature::unnamedAddr());

  // Switch any previous uses to the alias.
  if (Entry) {
    llvm_unreachable("NYI");
  } else {
    // Name already set by createCIRFunction
  }

  // Finally, set up the alias with its proper name and attributes.
  setCommonAttributes(AliasDecl, Alias);
  return false;
}

static void buildDeclInit(CIRGenFunction &CGF, const VarDecl *D,
                          Address DeclPtr) {
  assert((D->hasGlobalStorage() ||
          (D->hasLocalStorage() &&
           CGF.getContext().getLangOpts().OpenCLCPlusPlus)) &&
         "VarDecl must have global or local (in the case of OpenCL) storage!");
  assert(!D->getType()->isReferenceType() &&
         "Should not call buildDeclInit on a reference!");

  QualType type = D->getType();
  LValue lv = CGF.makeAddrLValue(DeclPtr, type);

  const Expr *Init = D->getInit();
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case TEK_Aggregate:
    CGF.buildAggExpr(
        Init, AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                      AggValueSlot::DoesNotNeedGCBarriers,
                                      AggValueSlot::IsNotAliased,
                                      AggValueSlot::DoesNotOverlap));
    return;
  case TEK_Scalar:
    CGF.buildScalarInit(Init, CGF.getLoc(D->getLocation()), lv, false);
    return;
  case TEK_Complex:
    llvm_unreachable("complext evaluation NYI");
  }
}

static void buildDeclDestory(CIRGenFunction &CGF, const VarDecl *D,
                             Address DeclPtr) {
  // Honor __attribute__((no_destroy)) and bail instead of attempting
  // to emit a reference to a possibly nonexistent destructor, which
  // in turn can cause a crash. This will result in a global constructor
  // that isn't balanced out by a destructor call as intended by the
  // attribute. This also checks for -fno-c++-static-destructors and
  // bails even if the attribute is not present.
  assert(D->needsDestruction(CGF.getContext()) == QualType::DK_cxx_destructor);

  auto &CGM = CGF.CGM;

  // If __cxa_atexit is disabled via a flag, a different helper function is
  // generated elsewhere which uses atexit instead, and it takes the destructor
  // directly.
  auto UsingExternalHelper = CGM.getCodeGenOpts().CXAAtExit;
  QualType type = D->getType();
  const CXXRecordDecl *Record = type->getAsCXXRecordDecl();
  bool CanRegisterDestructor =
      Record && (!CGM.getCXXABI().HasThisReturn(
                     GlobalDecl(Record->getDestructor(), Dtor_Complete)) ||
                 CGM.getCXXABI().canCallMismatchedFunctionType());
  if (Record && (CanRegisterDestructor || UsingExternalHelper)) {
    assert(!D->getTLSKind() && "TLS NYI");
    CXXDestructorDecl *Dtor = Record->getDestructor();
    CGM.getCXXABI().buildDestructorCall(CGF, Dtor, Dtor_Complete,
                                        /*ForVirtualBase=*/false,
                                        /*Delegating=*/false, DeclPtr, type);
  } else {
    llvm_unreachable("array destructors not yet supported!");
  }
}

mlir::cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl GD) {
  const auto &FnInfo = getTypes().arrangeCXXStructorDeclaration(GD);
  auto Fn = getAddrOfCXXStructor(GD, &FnInfo, /*FnType=*/nullptr,
                                 /*DontDefer=*/true, ForDefinition);

  setFunctionLinkage(GD, Fn);
  CIRGenFunction CGF{*this, builder};
  CurCGF = &CGF;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    CGF.generateCode(GD, Fn, FnInfo);
  }
  CurCGF = nullptr;

  // TODO: setNonAliasAttributes
  // TODO: SetLLVMFunctionAttributesForDefinition
  return Fn;
}

void CIRGenModule::codegenGlobalInitCxxStructor(const VarDecl *D,
                                                mlir::cir::GlobalOp Addr,
                                                bool NeedsCtor,
                                                bool NeedsDtor) {
  assert(D && " Expected a global declaration!");
  CIRGenFunction CGF{*this, builder, true};
  CurCGF = &CGF;
  CurCGF->CurFn = Addr;
  Addr.setAstAttr(mlir::cir::ASTVarDeclAttr::get(builder.getContext(), D));

  if (NeedsCtor) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto block = builder.createBlock(&Addr.getCtorRegion());
    builder.setInsertionPointToStart(block);
    Address DeclAddr(getAddrOfGlobalVar(D), getASTContext().getDeclAlign(D));
    buildDeclInit(CGF, D, DeclAddr);
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::cir::YieldOp>(Addr->getLoc());
  }

  if (NeedsDtor) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto block = builder.createBlock(&Addr.getDtorRegion());
    builder.setInsertionPointToStart(block);
    Address DeclAddr(getAddrOfGlobalVar(D), getASTContext().getDeclAlign(D));
    buildDeclDestory(CGF, D, DeclAddr);
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::cir::YieldOp>(Addr->getLoc());
  }

  CurCGF = nullptr;
}
