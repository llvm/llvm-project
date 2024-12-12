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
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include <cassert>

using namespace clang;
using namespace clang::CIRGen;

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
    assert(!cir::MissingFeatures::sanitizeDtor());

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
  const ASTRecordLayout &ClassLayout = astContext.getASTRecordLayout(Class);
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
  if (!cir::isValidLinkage(Linkage))
    return true;

  auto TargetLinkage = getFunctionLinkage(TargetDecl);

  // Check if we have it already.
  StringRef MangledName = getMangledName(AliasDecl);
  auto Entry = getGlobalValue(MangledName);
  auto globalValue = dyn_cast_or_null<cir::CIRGlobalValueInterface>(Entry);
  if (Entry && globalValue && !globalValue.isDeclaration())
    return false;
  if (Replacements.count(MangledName))
    return false;

  [[maybe_unused]] auto AliasValueType = getTypes().GetFunctionType(AliasDecl);

  // Find the referent.
  auto Aliasee = cast<cir::FuncOp>(GetAddrOfGlobal(TargetDecl));
  auto AliaseeGV = dyn_cast_or_null<cir::CIRGlobalValueInterface>(
      GetAddrOfGlobal(TargetDecl));
  // Instead of creating as alias to a linkonce_odr, replace all of the uses
  // of the aliasee.
  if (cir::isDiscardableIfUnused(Linkage) &&
      !(TargetLinkage == cir::GlobalLinkageKind::AvailableExternallyLinkage &&
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
  if (cir::isWeakForLinker(Linkage) && getTriple().isOSBinFormatCOFF()) {
    llvm_unreachable("NYI");
  }

  // If we don't have a definition for the destructor yet or the definition
  // is
  // avaialable_externally, don't emit an alias.  We can't emit aliases to
  // declarations; that's just not how aliases work.
  if (AliaseeGV && AliaseeGV.isDeclarationForLinker())
    return true;

  // Don't create an alias to a linker weak symbol. This avoids producing
  // different COMDATs in different TUs. Another option would be to
  // output the alias both for weak_odr and linkonce_odr, but that
  // requires explicit comdat support in the IL.
  if (cir::isWeakForLinker(TargetLinkage))
    llvm_unreachable("NYI");

  // Create the alias with no name.
  emitAliasForGlobal(MangledName, Entry, AliasDecl, Aliasee, Linkage);
  return false;
}

static void emitDeclInit(CIRGenFunction &cgf, const VarDecl *varDecl,
                         Address declPtr) {
  assert((varDecl->hasGlobalStorage() ||
          (varDecl->hasLocalStorage() &&
           cgf.getContext().getLangOpts().OpenCLCPlusPlus)) &&
         "VarDecl must have global or local (in the case of OpenCL) storage!");
  assert(!varDecl->getType()->isReferenceType() &&
         "Should not call emitDeclInit on a reference!");

  QualType type = varDecl->getType();
  LValue lv = cgf.makeAddrLValue(declPtr, type);

  const Expr *init = varDecl->getInit();
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case cir::TEK_Scalar:
    if (lv.isObjCStrong())
      llvm_unreachable("NYI");
    else if (lv.isObjCWeak())
      llvm_unreachable("NYI");
    else
      cgf.emitScalarInit(init, cgf.getLoc(varDecl->getLocation()), lv, false);
    return;
  case cir::TEK_Complex:
    llvm_unreachable("complext evaluation NYI");
  case cir::TEK_Aggregate:
    cgf.emitAggExpr(init,
                    AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                            AggValueSlot::DoesNotNeedGCBarriers,
                                            AggValueSlot::IsNotAliased,
                                            AggValueSlot::DoesNotOverlap));
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

static void emitDeclDestroy(CIRGenFunction &CGF, const VarDecl *D) {
  // Honor __attribute__((no_destroy)) and bail instead of attempting
  // to emit a reference to a possibly nonexistent destructor, which
  // in turn can cause a crash. This will result in a global constructor
  // that isn't balanced out by a destructor call as intended by the
  // attribute. This also checks for -fno-c++-static-destructors and
  // bails even if the attribute is not present.
  QualType::DestructionKind DtorKind = D->needsDestruction(CGF.getContext());

  // FIXME:  __attribute__((cleanup)) ?

  switch (DtorKind) {
  case QualType::DK_none:
    return;

  case QualType::DK_cxx_destructor:
    break;

  case QualType::DK_objc_strong_lifetime:
  case QualType::DK_objc_weak_lifetime:
  case QualType::DK_nontrivial_c_struct:
    // We don't care about releasing objects during process teardown.
    assert(!D->getTLSKind() && "should have rejected this");
    return;
  }

  auto &CGM = CGF.CGM;
  QualType type = D->getType();

  // Special-case non-array C++ destructors, if they have the right signature.
  // Under some ABIs, destructors return this instead of void, and cannot be
  // passed directly to __cxa_atexit if the target does not allow this
  // mismatch.
  const CXXRecordDecl *Record = type->getAsCXXRecordDecl();
  bool CanRegisterDestructor =
      Record && (!CGM.getCXXABI().HasThisReturn(
                     GlobalDecl(Record->getDestructor(), Dtor_Complete)) ||
                 CGM.getCXXABI().canCallMismatchedFunctionType());

  // If __cxa_atexit is disabled via a flag, a different helper function is
  // generated elsewhere which uses atexit instead, and it takes the destructor
  // directly.
  auto UsingExternalHelper = CGM.getCodeGenOpts().CXAAtExit;
  cir::FuncOp fnOp;
  if (Record && (CanRegisterDestructor || UsingExternalHelper)) {
    assert(!D->getTLSKind() && "TLS NYI");
    assert(!Record->hasTrivialDestructor());
    assert(!cir::MissingFeatures::openCLCXX());
    CXXDestructorDecl *Dtor = Record->getDestructor();
    // In LLVM OG codegen this is done in registerGlobalDtor, but CIRGen
    // relies on LoweringPrepare for further decoupling, so build the
    // call right here.
    auto GD = GlobalDecl(Dtor, Dtor_Complete);
    auto structorInfo = CGM.getAddrAndTypeOfCXXStructor(GD);
    fnOp = structorInfo.second;
    CGF.getBuilder().createCallOp(
        CGF.getLoc(D->getSourceRange()),
        mlir::FlatSymbolRefAttr::get(fnOp.getSymNameAttr()),
        mlir::ValueRange{CGF.CGM.getAddrOfGlobalVar(D)});
  } else {
    llvm_unreachable("array destructors not yet supported!");
  }
  assert(fnOp && "expected cir.func");
  CGM.getCXXABI().registerGlobalDtor(CGF, D, fnOp, nullptr);
}

cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl GD) {
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

  setNonAliasAttributes(GD, Fn);
  setCIRFunctionAttributesForDefinition(cast<CXXMethodDecl>(GD.getDecl()), Fn);
  return Fn;
}

/// Emit code to cause the variable at the given address to be considered as
/// constant from this point onwards.
static void emitDeclInvariant(CIRGenFunction &CGF, const VarDecl *D) {
  return CGF.emitInvariantStart(
      CGF.getContext().getTypeSizeInChars(D->getType()));
}

void CIRGenFunction::emitInvariantStart([[maybe_unused]] CharUnits Size) {
  // Do not emit the intrinsic if we're not optimizing.
  if (!CGM.getCodeGenOpts().OptimizationLevel)
    return;

  assert(!cir::MissingFeatures::createInvariantIntrinsic());
}

void CIRGenModule::emitCXXGlobalVarDeclInit(const VarDecl *varDecl,
                                            cir::GlobalOp addr,
                                            bool performInit) {
  const Expr *init = varDecl->getInit();
  QualType ty = varDecl->getType();

  // TODO: handle address space
  // The address space of a static local variable (DeclPtr) may be different
  // from the address space of the "this" argument of the constructor. In that
  // case, we need an addrspacecast before calling the constructor.
  //
  // struct StructWithCtor {
  //   __device__ StructWithCtor() {...}
  // };
  // __device__ void foo() {
  //   __shared__ StructWithCtor s;
  //   ...
  // }
  //
  // For example, in the above CUDA code, the static local variable s has a
  // "shared" address space qualifier, but the constructor of StructWithCtor
  // expects "this" in the "generic" address space.
  assert(!cir::MissingFeatures::addressSpace());

  assert(varDecl && " Expected a global declaration!");
  CIRGenFunction cgf{*this, builder, true};
  llvm::SaveAndRestore<CIRGenFunction *> savedCGF(CurCGF, &cgf);
  CurCGF->CurFn = addr;

  CIRGenFunction::SourceLocRAIIObject fnLoc{cgf,
                                            getLoc(varDecl->getLocation())};

  addr.setAstAttr(cir::ASTVarDeclAttr::get(&getMLIRContext(), varDecl));

  if (!ty->isReferenceType()) {
    if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd &&
        varDecl->hasAttr<OMPThreadPrivateDeclAttr>()) {
      llvm_unreachable("NYI");
    }

    bool needsDtor = varDecl->needsDestruction(getASTContext()) ==
                     QualType::DK_cxx_destructor;
    // PerformInit, constant store invariant / destroy handled below.
    if (performInit) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block = builder.createBlock(&addr.getCtorRegion());
      CIRGenFunction::LexicalScope lexScope{*CurCGF, addr.getLoc(),
                                            builder.getInsertionBlock()};
      lexScope.setAsGlobalInit();

      builder.setInsertionPointToStart(block);
      Address declAddr(getAddrOfGlobalVar(varDecl),
                       getASTContext().getDeclAlign(varDecl));
      emitDeclInit(cgf, varDecl, declAddr);
      builder.setInsertionPointToEnd(block);
      builder.create<cir::YieldOp>(addr->getLoc());
    }

    if (varDecl->getType().isConstantStorage(getASTContext(), true,
                                             !needsDtor)) {
      // TODO(CIR): this leads to a missing feature in the moment, probably also
      // need a LexicalScope to be inserted here.
      emitDeclInvariant(cgf, varDecl);
    } else {
      // If not constant storage we'll emit this regardless of NeedsDtor value.
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto *block = builder.createBlock(&addr.getDtorRegion());
      CIRGenFunction::LexicalScope lexScope{*CurCGF, addr.getLoc(),
                                            builder.getInsertionBlock()};
      lexScope.setAsGlobalInit();

      builder.setInsertionPointToStart(block);
      emitDeclDestroy(cgf, varDecl);
      builder.setInsertionPointToEnd(block);
      if (block->empty()) {
        block->erase();
        // Don't confuse lexical cleanup.
        builder.clearInsertionPoint();
      } else
        builder.create<cir::YieldOp>(addr->getLoc());
    }
    return;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto *block = builder.createBlock(&addr.getCtorRegion());
  CIRGenFunction::LexicalScope lexScope{*CurCGF, addr.getLoc(),
                                        builder.getInsertionBlock()};
  lexScope.setAsGlobalInit();
  builder.setInsertionPointToStart(block);
  auto getGlobal = builder.createGetGlobal(addr);

  Address declAddr(getGlobal, getGlobal.getType(),
                   getASTContext().getDeclAlign(varDecl));
  assert(performInit && "cannot have constant initializer which needs "
                        "destruction for reference");
  RValue rv = cgf.emitReferenceBindingToExpr(init);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Operation *rvalueDefOp = rv.getScalarVal().getDefiningOp();
    if (rvalueDefOp && rvalueDefOp->getBlock()) {
      mlir::Block *rvalSrcBlock = rvalueDefOp->getBlock();
      if (!rvalSrcBlock->empty() && isa<cir::YieldOp>(rvalSrcBlock->back())) {
        auto &front = rvalSrcBlock->front();
        getGlobal.getDefiningOp()->moveBefore(&front);
        auto yield = cast<cir::YieldOp>(rvalSrcBlock->back());
        builder.setInsertionPoint(yield);
      }
    }
    cgf.emitStoreOfScalar(rv.getScalarVal(), declAddr, false, ty);
  }
  builder.setInsertionPointToEnd(block);
  builder.create<cir::YieldOp>(addr->getLoc());
}
