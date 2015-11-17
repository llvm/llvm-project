//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGBlocks.h"
#include "CGCleanup.h"
#include "CGDebugInfo.h"
#include "CGOpenCLRuntime.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDecl(const Decl &D) {
  switch (D.getKind()) {
  case Decl::BuiltinTemplate:
  case Decl::TranslationUnit:
  case Decl::ExternCContext:
  case Decl::Namespace:
  case Decl::UnresolvedUsingTypename:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
  case Decl::VarTemplateSpecialization:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::TemplateTypeParm:
  case Decl::UnresolvedUsingValue:
  case Decl::NonTypeTemplateParm:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion:
  case Decl::Field:
  case Decl::MSProperty:
  case Decl::IndirectField:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:
  case Decl::ParmVar:
  case Decl::ImplicitParam:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::TemplateTemplateParm:
  case Decl::ObjCMethod:
  case Decl::ObjCCategory:
  case Decl::ObjCProtocol:
  case Decl::ObjCInterface:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
  case Decl::ObjCProperty:
  case Decl::ObjCCompatibleAlias:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::Friend:
  case Decl::FriendTemplate:
  case Decl::Block:
  case Decl::Captured:
  case Decl::ClassScopeFunctionSpecialization:
  case Decl::UsingShadow:
  case Decl::ObjCTypeParam:
    llvm_unreachable("Declaration should not be in declstmts!");
  case Decl::Function:  // void X();
  case Decl::Record:    // struct/union/class X;
  case Decl::Enum:      // enum X;
  case Decl::EnumConstant: // enum ? { X = ? }
  case Decl::CXXRecord: // struct/union/class X; [C++]
  case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
  case Decl::Label:        // __label__ x;
  case Decl::Import:
  case Decl::OMPThreadPrivate:
  case Decl::Empty:
    // None of these decls require codegen support.
    return;

  case Decl::NamespaceAlias:
    if (CGDebugInfo *DI = getDebugInfo())
        DI->EmitNamespaceAlias(cast<NamespaceAliasDecl>(D));
    return;
  case Decl::Using:          // using X; [C++]
    if (CGDebugInfo *DI = getDebugInfo())
        DI->EmitUsingDecl(cast<UsingDecl>(D));
    return;
  case Decl::UsingDirective: // using namespace X; [C++]
    if (CGDebugInfo *DI = getDebugInfo())
      DI->EmitUsingDirective(cast<UsingDirectiveDecl>(D));
    return;
  case Decl::Var: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    return EmitVarDecl(VD);
  }

  case Decl::Typedef:      // typedef int X;
  case Decl::TypeAlias: {  // using X = int; [C++0x]
    const TypedefNameDecl &TD = cast<TypedefNameDecl>(D);
    QualType Ty = TD.getUnderlyingType();

    if (Ty->isVariablyModifiedType())
      EmitVariablyModifiedType(Ty);
  }
  }
}

/// EmitVarDecl - This method handles emission of any variable declaration
/// inside a function, including static vars etc.
void CodeGenFunction::EmitVarDecl(const VarDecl &D) {
  if (D.isStaticLocal()) {
    llvm::GlobalValue::LinkageTypes Linkage =
        CGM.getLLVMLinkageVarDefinition(&D, /*isConstant=*/false);

    // FIXME: We need to force the emission/use of a guard variable for
    // some variables even if we can constant-evaluate them because
    // we can't guarantee every translation unit will constant-evaluate them.

    return EmitStaticVarDecl(D, Linkage);
  }

  if (D.hasExternalStorage())
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;

  if (D.getType().getAddressSpace() == LangAS::opencl_local)
    return CGM.getOpenCLRuntime().EmitWorkGroupLocalVarDecl(*this, D);

  assert(D.hasLocalStorage());
  return EmitAutoVarDecl(D);
}

static std::string getStaticDeclName(CodeGenModule &CGM, const VarDecl &D) {
  if (CGM.getLangOpts().CPlusPlus)
    return CGM.getMangledName(&D).str();

  // If this isn't C++, we don't need a mangled name, just a pretty one.
  assert(!D.isExternallyVisible() && "name shouldn't matter");
  std::string ContextName;
  const DeclContext *DC = D.getDeclContext();
  if (auto *CD = dyn_cast<CapturedDecl>(DC))
    DC = cast<DeclContext>(CD->getNonClosureContext());
  if (const auto *FD = dyn_cast<FunctionDecl>(DC))
    ContextName = CGM.getMangledName(FD);
  else if (const auto *BD = dyn_cast<BlockDecl>(DC))
    ContextName = CGM.getBlockMangledName(GlobalDecl(), BD);
  else if (const auto *OMD = dyn_cast<ObjCMethodDecl>(DC))
    ContextName = OMD->getSelector().getAsString();
  else
    llvm_unreachable("Unknown context for static var decl");

  ContextName += "." + D.getNameAsString();
  return ContextName;
}

llvm::Constant *CodeGenModule::getOrCreateStaticVarDecl(
    const VarDecl &D, llvm::GlobalValue::LinkageTypes Linkage) {
  // In general, we don't always emit static var decls once before we reference
  // them. It is possible to reference them before emitting the function that
  // contains them, and it is possible to emit the containing function multiple
  // times.
  if (llvm::Constant *ExistingGV = StaticLocalDeclMap[&D])
    return ExistingGV;

  QualType Ty = D.getType();
  assert(Ty->isConstantSizeType() && "VLAs can't be static");

  // Use the label if the variable is renamed with the asm-label extension.
  std::string Name;
  if (D.hasAttr<AsmLabelAttr>())
    Name = getMangledName(&D);
  else
    Name = getStaticDeclName(*this, D);

  llvm::Type *LTy = getTypes().ConvertTypeForMem(Ty);
  unsigned AddrSpace =
      GetGlobalVarAddressSpace(&D, getContext().getTargetAddressSpace(Ty));

  // Local address space cannot have an initializer.
  llvm::Constant *Init = nullptr;
  if (Ty.getAddressSpace() != LangAS::opencl_local)
    Init = EmitNullConstant(Ty);
  else
    Init = llvm::UndefValue::get(LTy);

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), LTy,
                             Ty.isConstant(getContext()), Linkage,
                             Init, Name, nullptr,
                             llvm::GlobalVariable::NotThreadLocal,
                             AddrSpace);
  GV->setAlignment(getContext().getDeclAlign(&D).getQuantity());
  setGlobalVisibility(GV, &D);

  if (supportsCOMDAT() && GV->isWeakForLinker())
    GV->setComdat(TheModule.getOrInsertComdat(GV->getName()));

  if (D.getTLSKind())
    setTLSMode(GV, D);

  if (D.isExternallyVisible()) {
    if (D.hasAttr<DLLImportAttr>())
      GV->setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
    else if (D.hasAttr<DLLExportAttr>())
      GV->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  }

  // Make sure the result is of the correct type.
  unsigned ExpectedAddrSpace = getContext().getTargetAddressSpace(Ty);
  llvm::Constant *Addr = GV;
  if (AddrSpace != ExpectedAddrSpace) {
    llvm::PointerType *PTy = llvm::PointerType::get(LTy, ExpectedAddrSpace);
    Addr = llvm::ConstantExpr::getAddrSpaceCast(GV, PTy);
  }

  setStaticLocalDeclAddress(&D, Addr);

  // Ensure that the static local gets initialized by making sure the parent
  // function gets emitted eventually.
  const Decl *DC = cast<Decl>(D.getDeclContext());

  // We can't name blocks or captured statements directly, so try to emit their
  // parents.
  if (isa<BlockDecl>(DC) || isa<CapturedDecl>(DC)) {
    DC = DC->getNonClosureContext();
    // FIXME: Ensure that global blocks get emitted.
    if (!DC)
      return Addr;
  }

  GlobalDecl GD;
  if (const auto *CD = dyn_cast<CXXConstructorDecl>(DC))
    GD = GlobalDecl(CD, Ctor_Base);
  else if (const auto *DD = dyn_cast<CXXDestructorDecl>(DC))
    GD = GlobalDecl(DD, Dtor_Base);
  else if (const auto *FD = dyn_cast<FunctionDecl>(DC))
    GD = GlobalDecl(FD);
  else {
    // Don't do anything for Obj-C method decls or global closures. We should
    // never defer them.
    assert(isa<ObjCMethodDecl>(DC) && "unexpected parent code decl");
  }
  if (GD.getDecl())
    (void)GetAddrOfGlobal(GD);

  return Addr;
}

/// hasNontrivialDestruction - Determine whether a type's destruction is
/// non-trivial. If so, and the variable uses static initialization, we must
/// register its destructor to run on exit.
static bool hasNontrivialDestruction(QualType T) {
  CXXRecordDecl *RD = T->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
  return RD && !RD->hasTrivialDestructor();
}

/// AddInitializerToStaticVarDecl - Add the initializer for 'D' to the
/// global variable that has already been created for it.  If the initializer
/// has a different type than GV does, this may free GV and return a different
/// one.  Otherwise it just returns GV.
llvm::GlobalVariable *
CodeGenFunction::AddInitializerToStaticVarDecl(const VarDecl &D,
                                               llvm::GlobalVariable *GV) {
  llvm::Constant *Init = CGM.EmitConstantInit(D, this);

  // If constant emission failed, then this should be a C++ static
  // initializer.
  if (!Init) {
    if (!getLangOpts().CPlusPlus)
      CGM.ErrorUnsupported(D.getInit(), "constant l-value expression");
    else if (Builder.GetInsertBlock()) {
      // Since we have a static initializer, this global variable can't
      // be constant.
      GV->setConstant(false);

      EmitCXXGuardedInit(D, GV, /*PerformInit*/true);
    }
    return GV;
  }

  // The initializer may differ in type from the global. Rewrite
  // the global to match the initializer.  (We have to do this
  // because some types, like unions, can't be completely represented
  // in the LLVM type system.)
  if (GV->getType()->getElementType() != Init->getType()) {
    llvm::GlobalVariable *OldGV = GV;

    GV = new llvm::GlobalVariable(CGM.getModule(), Init->getType(),
                                  OldGV->isConstant(),
                                  OldGV->getLinkage(), Init, "",
                                  /*InsertBefore*/ OldGV,
                                  OldGV->getThreadLocalMode(),
                           CGM.getContext().getTargetAddressSpace(D.getType()));
    GV->setVisibility(OldGV->getVisibility());
    GV->setComdat(OldGV->getComdat());

    // Steal the name of the old global
    GV->takeName(OldGV);

    // Replace all uses of the old global with the new global
    llvm::Constant *NewPtrForOldDecl =
    llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
    OldGV->replaceAllUsesWith(NewPtrForOldDecl);

    // Erase the old global, since it is no longer used.
    OldGV->eraseFromParent();
  }

  GV->setConstant(CGM.isTypeConstant(D.getType(), true));
  GV->setInitializer(Init);

  if (hasNontrivialDestruction(D.getType())) {
    // We have a constant initializer, but a nontrivial destructor. We still
    // need to perform a guarded "initialization" in order to register the
    // destructor.
    EmitCXXGuardedInit(D, GV, /*PerformInit*/false);
  }

  return GV;
}

void CodeGenFunction::EmitStaticVarDecl(const VarDecl &D,
                                      llvm::GlobalValue::LinkageTypes Linkage) {
  // Check to see if we already have a global variable for this
  // declaration.  This can happen when double-emitting function
  // bodies, e.g. with complete and base constructors.
  llvm::Constant *addr = CGM.getOrCreateStaticVarDecl(D, Linkage);
  CharUnits alignment = getContext().getDeclAlign(&D);

  // Store into LocalDeclMap before generating initializer to handle
  // circular references.
  setAddrOfLocalVar(&D, Address(addr, alignment));

  // We can't have a VLA here, but we can have a pointer to a VLA,
  // even though that doesn't really make any sense.
  // Make sure to evaluate VLA bounds now so that we have them for later.
  if (D.getType()->isVariablyModifiedType())
    EmitVariablyModifiedType(D.getType());

  // Save the type in case adding the initializer forces a type change.
  llvm::Type *expectedType = addr->getType();

  llvm::GlobalVariable *var =
    cast<llvm::GlobalVariable>(addr->stripPointerCasts());
  // If this value has an initializer, emit it.
  if (D.getInit())
    var = AddInitializerToStaticVarDecl(D, var);

  var->setAlignment(alignment.getQuantity());

  if (D.hasAttr<AnnotateAttr>())
    CGM.AddGlobalAnnotations(&D, var);

  if (const SectionAttr *SA = D.getAttr<SectionAttr>())
    var->setSection(SA->getName());

  if (D.hasAttr<UsedAttr>())
    CGM.addUsedGlobal(var);

  // We may have to cast the constant because of the initializer
  // mismatch above.
  //
  // FIXME: It is really dangerous to store this in the map; if anyone
  // RAUW's the GV uses of this constant will be invalid.
  llvm::Constant *castedAddr =
    llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(var, expectedType);
  if (var != castedAddr)
    LocalDeclMap.find(&D)->second = Address(castedAddr, alignment);
  CGM.setStaticLocalDeclAddress(&D, castedAddr);

  CGM.getSanitizerMetadata()->reportGlobalToASan(var, D);

  // Emit global variable debug descriptor for static vars.
  CGDebugInfo *DI = getDebugInfo();
  if (DI &&
      CGM.getCodeGenOpts().getDebugInfo() >= CodeGenOptions::LimitedDebugInfo) {
    DI->setLocation(D.getLocation());
    DI->EmitGlobalVariable(var, &D);
  }
}

namespace {
  struct DestroyObject final : EHScopeStack::Cleanup {
    DestroyObject(Address addr, QualType type,
                  CodeGenFunction::Destroyer *destroyer,
                  bool useEHCleanupForArray)
      : addr(addr), type(type), destroyer(destroyer),
        useEHCleanupForArray(useEHCleanupForArray) {}

    Address addr;
    QualType type;
    CodeGenFunction::Destroyer *destroyer;
    bool useEHCleanupForArray;

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      // Don't use an EH cleanup recursively from an EH cleanup.
      bool useEHCleanupForArray =
        flags.isForNormalCleanup() && this->useEHCleanupForArray;

      CGF.emitDestroy(addr, type, destroyer, useEHCleanupForArray);
    }
  };

  struct DestroyNRVOVariable final : EHScopeStack::Cleanup {
    DestroyNRVOVariable(Address addr,
                        const CXXDestructorDecl *Dtor,
                        llvm::Value *NRVOFlag)
      : Dtor(Dtor), NRVOFlag(NRVOFlag), Loc(addr) {}

    const CXXDestructorDecl *Dtor;
    llvm::Value *NRVOFlag;
    Address Loc;

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      // Along the exceptions path we always execute the dtor.
      bool NRVO = flags.isForNormalCleanup() && NRVOFlag;

      llvm::BasicBlock *SkipDtorBB = nullptr;
      if (NRVO) {
        // If we exited via NRVO, we skip the destructor call.
        llvm::BasicBlock *RunDtorBB = CGF.createBasicBlock("nrvo.unused");
        SkipDtorBB = CGF.createBasicBlock("nrvo.skipdtor");
        llvm::Value *DidNRVO =
          CGF.Builder.CreateFlagLoad(NRVOFlag, "nrvo.val");
        CGF.Builder.CreateCondBr(DidNRVO, SkipDtorBB, RunDtorBB);
        CGF.EmitBlock(RunDtorBB);
      }

      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                /*ForVirtualBase=*/false,
                                /*Delegating=*/false,
                                Loc);

      if (NRVO) CGF.EmitBlock(SkipDtorBB);
    }
  };

  struct CallStackRestore final : EHScopeStack::Cleanup {
    Address Stack;
    CallStackRestore(Address Stack) : Stack(Stack) {}
    void Emit(CodeGenFunction &CGF, Flags flags) override {
      llvm::Value *V = CGF.Builder.CreateLoad(Stack);
      llvm::Value *F = CGF.CGM.getIntrinsic(llvm::Intrinsic::stackrestore);
      CGF.Builder.CreateCall(F, V);
    }
  };

  struct ExtendGCLifetime final : EHScopeStack::Cleanup {
    const VarDecl &Var;
    ExtendGCLifetime(const VarDecl *var) : Var(*var) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      // Compute the address of the local variable, in case it's a
      // byref or something.
      DeclRefExpr DRE(const_cast<VarDecl*>(&Var), false,
                      Var.getType(), VK_LValue, SourceLocation());
      llvm::Value *value = CGF.EmitLoadOfScalar(CGF.EmitDeclRefLValue(&DRE),
                                                SourceLocation());
      CGF.EmitExtendGCLifetime(value);
    }
  };

  struct CallCleanupFunction final : EHScopeStack::Cleanup {
    llvm::Constant *CleanupFn;
    const CGFunctionInfo &FnInfo;
    const VarDecl &Var;

    CallCleanupFunction(llvm::Constant *CleanupFn, const CGFunctionInfo *Info,
                        const VarDecl *Var)
      : CleanupFn(CleanupFn), FnInfo(*Info), Var(*Var) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      DeclRefExpr DRE(const_cast<VarDecl*>(&Var), false,
                      Var.getType(), VK_LValue, SourceLocation());
      // Compute the address of the local variable, in case it's a byref
      // or something.
      llvm::Value *Addr = CGF.EmitDeclRefLValue(&DRE).getPointer();

      // In some cases, the type of the function argument will be different from
      // the type of the pointer. An example of this is
      // void f(void* arg);
      // __attribute__((cleanup(f))) void *g;
      //
      // To fix this we insert a bitcast here.
      QualType ArgTy = FnInfo.arg_begin()->type;
      llvm::Value *Arg =
        CGF.Builder.CreateBitCast(Addr, CGF.ConvertType(ArgTy));

      CallArgList Args;
      Args.add(RValue::get(Arg),
               CGF.getContext().getPointerType(Var.getType()));
      CGF.EmitCall(FnInfo, CleanupFn, ReturnValueSlot(), Args);
    }
  };

  /// A cleanup to call @llvm.lifetime.end.
  class CallLifetimeEnd final : public EHScopeStack::Cleanup {
    llvm::Value *Addr;
    llvm::Value *Size;
  public:
    CallLifetimeEnd(Address addr, llvm::Value *size)
      : Addr(addr.getPointer()), Size(size) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      CGF.EmitLifetimeEnd(Size, Addr);
    }
  };
}

/// EmitAutoVarWithLifetime - Does the setup required for an automatic
/// variable with lifetime.
static void EmitAutoVarWithLifetime(CodeGenFunction &CGF, const VarDecl &var,
                                    Address addr,
                                    Qualifiers::ObjCLifetime lifetime) {
  switch (lifetime) {
  case Qualifiers::OCL_None:
    llvm_unreachable("present but none");

  case Qualifiers::OCL_ExplicitNone:
    // nothing to do
    break;

  case Qualifiers::OCL_Strong: {
    CodeGenFunction::Destroyer *destroyer =
      (var.hasAttr<ObjCPreciseLifetimeAttr>()
       ? CodeGenFunction::destroyARCStrongPrecise
       : CodeGenFunction::destroyARCStrongImprecise);

    CleanupKind cleanupKind = CGF.getARCCleanupKind();
    CGF.pushDestroy(cleanupKind, addr, var.getType(), destroyer,
                    cleanupKind & EHCleanup);
    break;
  }
  case Qualifiers::OCL_Autoreleasing:
    // nothing to do
    break;

  case Qualifiers::OCL_Weak:
    // __weak objects always get EH cleanups; otherwise, exceptions
    // could cause really nasty crashes instead of mere leaks.
    CGF.pushDestroy(NormalAndEHCleanup, addr, var.getType(),
                    CodeGenFunction::destroyARCWeak,
                    /*useEHCleanup*/ true);
    break;
  }
}

static bool isAccessedBy(const VarDecl &var, const Stmt *s) {
  if (const Expr *e = dyn_cast<Expr>(s)) {
    // Skip the most common kinds of expressions that make
    // hierarchy-walking expensive.
    s = e = e->IgnoreParenCasts();

    if (const DeclRefExpr *ref = dyn_cast<DeclRefExpr>(e))
      return (ref->getDecl() == &var);
    if (const BlockExpr *be = dyn_cast<BlockExpr>(e)) {
      const BlockDecl *block = be->getBlockDecl();
      for (const auto &I : block->captures()) {
        if (I.getVariable() == &var)
          return true;
      }
    }
  }

  for (const Stmt *SubStmt : s->children())
    // SubStmt might be null; as in missing decl or conditional of an if-stmt.
    if (SubStmt && isAccessedBy(var, SubStmt))
      return true;

  return false;
}

static bool isAccessedBy(const ValueDecl *decl, const Expr *e) {
  if (!decl) return false;
  if (!isa<VarDecl>(decl)) return false;
  const VarDecl *var = cast<VarDecl>(decl);
  return isAccessedBy(*var, e);
}

static bool tryEmitARCCopyWeakInit(CodeGenFunction &CGF,
                                   const LValue &destLV, const Expr *init) {
  bool needsCast = false;

  while (auto castExpr = dyn_cast<CastExpr>(init->IgnoreParens())) {
    switch (castExpr->getCastKind()) {
    // Look through casts that don't require representation changes.
    case CK_NoOp:
    case CK_BitCast:
    case CK_BlockPointerToObjCPointerCast:
      needsCast = true;
      break;

    // If we find an l-value to r-value cast from a __weak variable,
    // emit this operation as a copy or move.
    case CK_LValueToRValue: {
      const Expr *srcExpr = castExpr->getSubExpr();
      if (srcExpr->getType().getObjCLifetime() != Qualifiers::OCL_Weak)
        return false;

      // Emit the source l-value.
      LValue srcLV = CGF.EmitLValue(srcExpr);

      // Handle a formal type change to avoid asserting.
      auto srcAddr = srcLV.getAddress();
      if (needsCast) {
        srcAddr = CGF.Builder.CreateElementBitCast(srcAddr,
                                         destLV.getAddress().getElementType());
      }

      // If it was an l-value, use objc_copyWeak.
      if (srcExpr->getValueKind() == VK_LValue) {
        CGF.EmitARCCopyWeak(destLV.getAddress(), srcAddr);
      } else {
        assert(srcExpr->getValueKind() == VK_XValue);
        CGF.EmitARCMoveWeak(destLV.getAddress(), srcAddr);
      }
      return true;
    }

    // Stop at anything else.
    default:
      return false;
    }

    init = castExpr->getSubExpr();
    continue;
  }
  return false;
}

static void drillIntoBlockVariable(CodeGenFunction &CGF,
                                   LValue &lvalue,
                                   const VarDecl *var) {
  lvalue.setAddress(CGF.emitBlockByrefAddress(lvalue.getAddress(), var));
}

void CodeGenFunction::EmitScalarInit(const Expr *init, const ValueDecl *D,
                                     LValue lvalue, bool capturedByInit) {
  Qualifiers::ObjCLifetime lifetime = lvalue.getObjCLifetime();
  if (!lifetime) {
    llvm::Value *value = EmitScalarExpr(init);
    if (capturedByInit)
      drillIntoBlockVariable(*this, lvalue, cast<VarDecl>(D));
    EmitStoreThroughLValue(RValue::get(value), lvalue, true);
    return;
  }
  
  if (const CXXDefaultInitExpr *DIE = dyn_cast<CXXDefaultInitExpr>(init))
    init = DIE->getExpr();
    
  // If we're emitting a value with lifetime, we have to do the
  // initialization *before* we leave the cleanup scopes.
  if (const ExprWithCleanups *ewc = dyn_cast<ExprWithCleanups>(init)) {
    enterFullExpression(ewc);
    init = ewc->getSubExpr();
  }
  CodeGenFunction::RunCleanupsScope Scope(*this);

  // We have to maintain the illusion that the variable is
  // zero-initialized.  If the variable might be accessed in its
  // initializer, zero-initialize before running the initializer, then
  // actually perform the initialization with an assign.
  bool accessedByInit = false;
  if (lifetime != Qualifiers::OCL_ExplicitNone)
    accessedByInit = (capturedByInit || isAccessedBy(D, init));
  if (accessedByInit) {
    LValue tempLV = lvalue;
    // Drill down to the __block object if necessary.
    if (capturedByInit) {
      // We can use a simple GEP for this because it can't have been
      // moved yet.
      tempLV.setAddress(emitBlockByrefAddress(tempLV.getAddress(),
                                              cast<VarDecl>(D),
                                              /*follow*/ false));
    }

    auto ty = cast<llvm::PointerType>(tempLV.getAddress().getElementType());
    llvm::Value *zero = llvm::ConstantPointerNull::get(ty);

    // If __weak, we want to use a barrier under certain conditions.
    if (lifetime == Qualifiers::OCL_Weak)
      EmitARCInitWeak(tempLV.getAddress(), zero);

    // Otherwise just do a simple store.
    else
      EmitStoreOfScalar(zero, tempLV, /* isInitialization */ true);
  }

  // Emit the initializer.
  llvm::Value *value = nullptr;

  switch (lifetime) {
  case Qualifiers::OCL_None:
    llvm_unreachable("present but none");

  case Qualifiers::OCL_ExplicitNone:
    // nothing to do
    value = EmitScalarExpr(init);
    break;

  case Qualifiers::OCL_Strong: {
    value = EmitARCRetainScalarExpr(init);
    break;
  }

  case Qualifiers::OCL_Weak: {
    // If it's not accessed by the initializer, try to emit the
    // initialization with a copy or move.
    if (!accessedByInit && tryEmitARCCopyWeakInit(*this, lvalue, init)) {
      return;
    }

    // No way to optimize a producing initializer into this.  It's not
    // worth optimizing for, because the value will immediately
    // disappear in the common case.
    value = EmitScalarExpr(init);

    if (capturedByInit) drillIntoBlockVariable(*this, lvalue, cast<VarDecl>(D));
    if (accessedByInit)
      EmitARCStoreWeak(lvalue.getAddress(), value, /*ignored*/ true);
    else
      EmitARCInitWeak(lvalue.getAddress(), value);
    return;
  }

  case Qualifiers::OCL_Autoreleasing:
    value = EmitARCRetainAutoreleaseScalarExpr(init);
    break;
  }

  if (capturedByInit) drillIntoBlockVariable(*this, lvalue, cast<VarDecl>(D));

  // If the variable might have been accessed by its initializer, we
  // might have to initialize with a barrier.  We have to do this for
  // both __weak and __strong, but __weak got filtered out above.
  if (accessedByInit && lifetime == Qualifiers::OCL_Strong) {
    llvm::Value *oldValue = EmitLoadOfScalar(lvalue, init->getExprLoc());
    EmitStoreOfScalar(value, lvalue, /* isInitialization */ true);
    EmitARCRelease(oldValue, ARCImpreciseLifetime);
    return;
  }

  EmitStoreOfScalar(value, lvalue, /* isInitialization */ true);
}

/// EmitScalarInit - Initialize the given lvalue with the given object.
void CodeGenFunction::EmitScalarInit(llvm::Value *init, LValue lvalue) {
  Qualifiers::ObjCLifetime lifetime = lvalue.getObjCLifetime();
  if (!lifetime)
    return EmitStoreThroughLValue(RValue::get(init), lvalue, true);

  switch (lifetime) {
  case Qualifiers::OCL_None:
    llvm_unreachable("present but none");

  case Qualifiers::OCL_ExplicitNone:
    // nothing to do
    break;

  case Qualifiers::OCL_Strong:
    init = EmitARCRetain(lvalue.getType(), init);
    break;

  case Qualifiers::OCL_Weak:
    // Initialize and then skip the primitive store.
    EmitARCInitWeak(lvalue.getAddress(), init);
    return;

  case Qualifiers::OCL_Autoreleasing:
    init = EmitARCRetainAutorelease(lvalue.getType(), init);
    break;
  }

  EmitStoreOfScalar(init, lvalue, /* isInitialization */ true);
}

/// canEmitInitWithFewStoresAfterMemset - Decide whether we can emit the
/// non-zero parts of the specified initializer with equal or fewer than
/// NumStores scalar stores.
static bool canEmitInitWithFewStoresAfterMemset(llvm::Constant *Init,
                                                unsigned &NumStores) {
  // Zero and Undef never requires any extra stores.
  if (isa<llvm::ConstantAggregateZero>(Init) ||
      isa<llvm::ConstantPointerNull>(Init) ||
      isa<llvm::UndefValue>(Init))
    return true;
  if (isa<llvm::ConstantInt>(Init) || isa<llvm::ConstantFP>(Init) ||
      isa<llvm::ConstantVector>(Init) || isa<llvm::BlockAddress>(Init) ||
      isa<llvm::ConstantExpr>(Init))
    return Init->isNullValue() || NumStores--;

  // See if we can emit each element.
  if (isa<llvm::ConstantArray>(Init) || isa<llvm::ConstantStruct>(Init)) {
    for (unsigned i = 0, e = Init->getNumOperands(); i != e; ++i) {
      llvm::Constant *Elt = cast<llvm::Constant>(Init->getOperand(i));
      if (!canEmitInitWithFewStoresAfterMemset(Elt, NumStores))
        return false;
    }
    return true;
  }
  
  if (llvm::ConstantDataSequential *CDS =
        dyn_cast<llvm::ConstantDataSequential>(Init)) {
    for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
      llvm::Constant *Elt = CDS->getElementAsConstant(i);
      if (!canEmitInitWithFewStoresAfterMemset(Elt, NumStores))
        return false;
    }
    return true;
  }

  // Anything else is hard and scary.
  return false;
}

/// emitStoresForInitAfterMemset - For inits that
/// canEmitInitWithFewStoresAfterMemset returned true for, emit the scalar
/// stores that would be required.
static void emitStoresForInitAfterMemset(llvm::Constant *Init, llvm::Value *Loc,
                                         bool isVolatile, CGBuilderTy &Builder) {
  assert(!Init->isNullValue() && !isa<llvm::UndefValue>(Init) &&
         "called emitStoresForInitAfterMemset for zero or undef value.");

  if (isa<llvm::ConstantInt>(Init) || isa<llvm::ConstantFP>(Init) ||
      isa<llvm::ConstantVector>(Init) || isa<llvm::BlockAddress>(Init) ||
      isa<llvm::ConstantExpr>(Init)) {
    Builder.CreateDefaultAlignedStore(Init, Loc, isVolatile);
    return;
  }
  
  if (llvm::ConstantDataSequential *CDS = 
        dyn_cast<llvm::ConstantDataSequential>(Init)) {
    for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
      llvm::Constant *Elt = CDS->getElementAsConstant(i);

      // If necessary, get a pointer to the element and emit it.
      if (!Elt->isNullValue() && !isa<llvm::UndefValue>(Elt))
        emitStoresForInitAfterMemset(
            Elt, Builder.CreateConstGEP2_32(Init->getType(), Loc, 0, i),
            isVolatile, Builder);
    }
    return;
  }

  assert((isa<llvm::ConstantStruct>(Init) || isa<llvm::ConstantArray>(Init)) &&
         "Unknown value type!");

  for (unsigned i = 0, e = Init->getNumOperands(); i != e; ++i) {
    llvm::Constant *Elt = cast<llvm::Constant>(Init->getOperand(i));

    // If necessary, get a pointer to the element and emit it.
    if (!Elt->isNullValue() && !isa<llvm::UndefValue>(Elt))
      emitStoresForInitAfterMemset(
          Elt, Builder.CreateConstGEP2_32(Init->getType(), Loc, 0, i),
          isVolatile, Builder);
  }
}


/// shouldUseMemSetPlusStoresToInitialize - Decide whether we should use memset
/// plus some stores to initialize a local variable instead of using a memcpy
/// from a constant global.  It is beneficial to use memset if the global is all
/// zeros, or mostly zeros and large.
static bool shouldUseMemSetPlusStoresToInitialize(llvm::Constant *Init,
                                                  uint64_t GlobalSize) {
  // If a global is all zeros, always use a memset.
  if (isa<llvm::ConstantAggregateZero>(Init)) return true;

  // If a non-zero global is <= 32 bytes, always use a memcpy.  If it is large,
  // do it if it will require 6 or fewer scalar stores.
  // TODO: Should budget depends on the size?  Avoiding a large global warrants
  // plopping in more stores.
  unsigned StoreBudget = 6;
  uint64_t SizeLimit = 32;

  return GlobalSize > SizeLimit &&
         canEmitInitWithFewStoresAfterMemset(Init, StoreBudget);
}

/// EmitAutoVarDecl - Emit code and set up an entry in LocalDeclMap for a
/// variable declaration with auto, register, or no storage class specifier.
/// These turn into simple stack objects, or GlobalValues depending on target.
void CodeGenFunction::EmitAutoVarDecl(const VarDecl &D) {
  AutoVarEmission emission = EmitAutoVarAlloca(D);
  EmitAutoVarInit(emission);
  EmitAutoVarCleanups(emission);
}

/// Emit a lifetime.begin marker if some criteria are satisfied.
/// \return a pointer to the temporary size Value if a marker was emitted, null
/// otherwise
llvm::Value *CodeGenFunction::EmitLifetimeStart(uint64_t Size,
                                                llvm::Value *Addr) {
  // For now, only in optimized builds.
  if (CGM.getCodeGenOpts().OptimizationLevel == 0)
    return nullptr;

  // Disable lifetime markers in msan builds.
  // FIXME: Remove this when msan works with lifetime markers.
  if (getLangOpts().Sanitize.has(SanitizerKind::Memory))
    return nullptr;

  llvm::Value *SizeV = llvm::ConstantInt::get(Int64Ty, Size);
  Addr = Builder.CreateBitCast(Addr, Int8PtrTy);
  llvm::CallInst *C =
      Builder.CreateCall(CGM.getLLVMLifetimeStartFn(), {SizeV, Addr});
  C->setDoesNotThrow();
  return SizeV;
}

void CodeGenFunction::EmitLifetimeEnd(llvm::Value *Size, llvm::Value *Addr) {
  Addr = Builder.CreateBitCast(Addr, Int8PtrTy);
  llvm::CallInst *C =
      Builder.CreateCall(CGM.getLLVMLifetimeEndFn(), {Size, Addr});
  C->setDoesNotThrow();
}

/// EmitAutoVarAlloca - Emit the alloca and debug information for a
/// local variable.  Does not emit initialization or destruction.
CodeGenFunction::AutoVarEmission
CodeGenFunction::EmitAutoVarAlloca(const VarDecl &D) {
  QualType Ty = D.getType();

  AutoVarEmission emission(D);

  bool isByRef = D.hasAttr<BlocksAttr>();
  emission.IsByRef = isByRef;

  CharUnits alignment = getContext().getDeclAlign(&D);

  // If the type is variably-modified, emit all the VLA sizes for it.
  if (Ty->isVariablyModifiedType())
    EmitVariablyModifiedType(Ty);

  Address address = Address::invalid();
  if (Ty->isConstantSizeType()) {
    bool NRVO = getLangOpts().ElideConstructors &&
      D.isNRVOVariable();

    // If this value is an array or struct with a statically determinable
    // constant initializer, there are optimizations we can do.
    //
    // TODO: We should constant-evaluate the initializer of any variable,
    // as long as it is initialized by a constant expression. Currently,
    // isConstantInitializer produces wrong answers for structs with
    // reference or bitfield members, and a few other cases, and checking
    // for POD-ness protects us from some of these.
    if (D.getInit() && (Ty->isArrayType() || Ty->isRecordType()) &&
        (D.isConstexpr() ||
         ((Ty.isPODType(getContext()) ||
           getContext().getBaseElementType(Ty)->isObjCObjectPointerType()) &&
          D.getInit()->isConstantInitializer(getContext(), false)))) {

      // If the variable's a const type, and it's neither an NRVO
      // candidate nor a __block variable and has no mutable members,
      // emit it as a global instead.
      if (CGM.getCodeGenOpts().MergeAllConstants && !NRVO && !isByRef &&
          CGM.isTypeConstant(Ty, true)) {
        EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);

        // Signal this condition to later callbacks.
        emission.Addr = Address::invalid();
        assert(emission.wasEmittedAsGlobal());
        return emission;
      }

      // Otherwise, tell the initialization code that we're in this case.
      emission.IsConstantAggregate = true;
    }

    // A normal fixed sized variable becomes an alloca in the entry block,
    // unless it's an NRVO variable.

    if (NRVO) {
      // The named return value optimization: allocate this variable in the
      // return slot, so that we can elide the copy when returning this
      // variable (C++0x [class.copy]p34).
      address = ReturnValue;

      if (const RecordType *RecordTy = Ty->getAs<RecordType>()) {
        if (!cast<CXXRecordDecl>(RecordTy->getDecl())->hasTrivialDestructor()) {
          // Create a flag that is used to indicate when the NRVO was applied
          // to this variable. Set it to zero to indicate that NRVO was not
          // applied.
          llvm::Value *Zero = Builder.getFalse();
          Address NRVOFlag =
            CreateTempAlloca(Zero->getType(), CharUnits::One(), "nrvo");
          EnsureInsertPoint();
          Builder.CreateStore(Zero, NRVOFlag);

          // Record the NRVO flag for this variable.
          NRVOFlags[&D] = NRVOFlag.getPointer();
          emission.NRVOFlag = NRVOFlag.getPointer();
        }
      }
    } else {
      CharUnits allocaAlignment;
      llvm::Type *allocaTy;
      if (isByRef) {
        auto &byrefInfo = getBlockByrefInfo(&D);
        allocaTy = byrefInfo.Type;
        allocaAlignment = byrefInfo.ByrefAlignment;
      } else {
        allocaTy = ConvertTypeForMem(Ty);
        allocaAlignment = alignment;
      }

      // Create the alloca.  Note that we set the name separately from
      // building the instruction so that it's there even in no-asserts
      // builds.
      address = CreateTempAlloca(allocaTy, allocaAlignment);
      address.getPointer()->setName(D.getName());

      // Don't emit lifetime markers for MSVC catch parameters. The lifetime of
      // the catch parameter starts in the catchpad instruction, and we can't
      // insert code in those basic blocks.
      bool IsMSCatchParam =
          D.isExceptionVariable() && getTarget().getCXXABI().isMicrosoft();

      // Emit a lifetime intrinsic if meaningful.  There's no point
      // in doing this if we don't have a valid insertion point (?).
      if (HaveInsertPoint() && !IsMSCatchParam) {
        uint64_t size = CGM.getDataLayout().getTypeAllocSize(allocaTy);
        emission.SizeForLifetimeMarkers =
          EmitLifetimeStart(size, address.getPointer());
      } else {
        assert(!emission.useLifetimeMarkers());
      }
    }
  } else {
    EnsureInsertPoint();

    if (!DidCallStackSave) {
      // Save the stack.
      Address Stack =
        CreateTempAlloca(Int8PtrTy, getPointerAlign(), "saved_stack");

      llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::stacksave);
      llvm::Value *V = Builder.CreateCall(F);
      Builder.CreateStore(V, Stack);

      DidCallStackSave = true;

      // Push a cleanup block and restore the stack there.
      // FIXME: in general circumstances, this should be an EH cleanup.
      pushStackRestore(NormalCleanup, Stack);
    }

    llvm::Value *elementCount;
    QualType elementType;
    std::tie(elementCount, elementType) = getVLASize(Ty);

    llvm::Type *llvmTy = ConvertTypeForMem(elementType);

    // Allocate memory for the array.
    llvm::AllocaInst *vla = Builder.CreateAlloca(llvmTy, elementCount, "vla");
    vla->setAlignment(alignment.getQuantity());

    address = Address(vla, alignment);
  }

  setAddrOfLocalVar(&D, address);
  emission.Addr = address;

  // Emit debug info for local var declaration.
  if (HaveInsertPoint())
    if (CGDebugInfo *DI = getDebugInfo()) {
      if (CGM.getCodeGenOpts().getDebugInfo()
            >= CodeGenOptions::LimitedDebugInfo) {
        DI->setLocation(D.getLocation());
        DI->EmitDeclareOfAutoVariable(&D, address.getPointer(), Builder);
      }
    }

  if (D.hasAttr<AnnotateAttr>())
    EmitVarAnnotations(&D, address.getPointer());

  return emission;
}

/// Determines whether the given __block variable is potentially
/// captured by the given expression.
static bool isCapturedBy(const VarDecl &var, const Expr *e) {
  // Skip the most common kinds of expressions that make
  // hierarchy-walking expensive.
  e = e->IgnoreParenCasts();

  if (const BlockExpr *be = dyn_cast<BlockExpr>(e)) {
    const BlockDecl *block = be->getBlockDecl();
    for (const auto &I : block->captures()) {
      if (I.getVariable() == &var)
        return true;
    }

    // No need to walk into the subexpressions.
    return false;
  }

  if (const StmtExpr *SE = dyn_cast<StmtExpr>(e)) {
    const CompoundStmt *CS = SE->getSubStmt();
    for (const auto *BI : CS->body())
      if (const auto *E = dyn_cast<Expr>(BI)) {
        if (isCapturedBy(var, E))
            return true;
      }
      else if (const auto *DS = dyn_cast<DeclStmt>(BI)) {
          // special case declarations
          for (const auto *I : DS->decls()) {
              if (const auto *VD = dyn_cast<VarDecl>((I))) {
                const Expr *Init = VD->getInit();
                if (Init && isCapturedBy(var, Init))
                  return true;
              }
          }
      }
      else
        // FIXME. Make safe assumption assuming arbitrary statements cause capturing.
        // Later, provide code to poke into statements for capture analysis.
        return true;
    return false;
  }

  for (const Stmt *SubStmt : e->children())
    if (isCapturedBy(var, cast<Expr>(SubStmt)))
      return true;

  return false;
}

/// \brief Determine whether the given initializer is trivial in the sense
/// that it requires no code to be generated.
bool CodeGenFunction::isTrivialInitializer(const Expr *Init) {
  if (!Init)
    return true;

  if (const CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init))
    if (CXXConstructorDecl *Constructor = Construct->getConstructor())
      if (Constructor->isTrivial() &&
          Constructor->isDefaultConstructor() &&
          !Construct->requiresZeroInitialization())
        return true;

  return false;
}
void CodeGenFunction::EmitAutoVarInit(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal()) return;

  const VarDecl &D = *emission.Variable;
  auto DL = ApplyDebugLocation::CreateDefaultArtificial(*this, D.getLocation());
  QualType type = D.getType();

  // If this local has an initializer, emit it now.
  const Expr *Init = D.getInit();

  // If we are at an unreachable point, we don't need to emit the initializer
  // unless it contains a label.
  if (!HaveInsertPoint()) {
    if (!Init || !ContainsLabel(Init)) return;
    EnsureInsertPoint();
  }

  // Initialize the structure of a __block variable.
  if (emission.IsByRef)
    emitByrefStructureInit(emission);

  if (isTrivialInitializer(Init))
    return;

  // Check whether this is a byref variable that's potentially
  // captured and moved by its own initializer.  If so, we'll need to
  // emit the initializer first, then copy into the variable.
  bool capturedByInit = emission.IsByRef && isCapturedBy(D, Init);

  Address Loc =
    capturedByInit ? emission.Addr : emission.getObjectAddress(*this);

  llvm::Constant *constant = nullptr;
  if (emission.IsConstantAggregate || D.isConstexpr()) {
    assert(!capturedByInit && "constant init contains a capturing block?");
    constant = CGM.EmitConstantInit(D, this);
  }

  if (!constant) {
    LValue lv = MakeAddrLValue(Loc, type);
    lv.setNonGC(true);
    return EmitExprAsInit(Init, &D, lv, capturedByInit);
  }

  if (!emission.IsConstantAggregate) {
    // For simple scalar/complex initialization, store the value directly.
    LValue lv = MakeAddrLValue(Loc, type);
    lv.setNonGC(true);
    return EmitStoreThroughLValue(RValue::get(constant), lv, true);
  }

  // If this is a simple aggregate initialization, we can optimize it
  // in various ways.
  bool isVolatile = type.isVolatileQualified();

  llvm::Value *SizeVal =
    llvm::ConstantInt::get(IntPtrTy,
                           getContext().getTypeSizeInChars(type).getQuantity());

  llvm::Type *BP = Int8PtrTy;
  if (Loc.getType() != BP)
    Loc = Builder.CreateBitCast(Loc, BP);

  // If the initializer is all or mostly zeros, codegen with memset then do
  // a few stores afterward.
  if (shouldUseMemSetPlusStoresToInitialize(constant,
                CGM.getDataLayout().getTypeAllocSize(constant->getType()))) {
    Builder.CreateMemSet(Loc, llvm::ConstantInt::get(Int8Ty, 0), SizeVal,
                         isVolatile);
    // Zero and undef don't require a stores.
    if (!constant->isNullValue() && !isa<llvm::UndefValue>(constant)) {
      Loc = Builder.CreateBitCast(Loc, constant->getType()->getPointerTo());
      emitStoresForInitAfterMemset(constant, Loc.getPointer(),
                                   isVolatile, Builder);
    }
  } else {
    // Otherwise, create a temporary global with the initializer then
    // memcpy from the global to the alloca.
    std::string Name = getStaticDeclName(CGM, D);
    llvm::GlobalVariable *GV =
      new llvm::GlobalVariable(CGM.getModule(), constant->getType(), true,
                               llvm::GlobalValue::PrivateLinkage,
                               constant, Name);
    GV->setAlignment(Loc.getAlignment().getQuantity());
    GV->setUnnamedAddr(true);

    Address SrcPtr = Address(GV, Loc.getAlignment());
    if (SrcPtr.getType() != BP)
      SrcPtr = Builder.CreateBitCast(SrcPtr, BP);

    Builder.CreateMemCpy(Loc, SrcPtr, SizeVal, isVolatile);
  }
}

/// Emit an expression as an initializer for a variable at the given
/// location.  The expression is not necessarily the normal
/// initializer for the variable, and the address is not necessarily
/// its normal location.
///
/// \param init the initializing expression
/// \param var the variable to act as if we're initializing
/// \param loc the address to initialize; its type is a pointer
///   to the LLVM mapping of the variable's type
/// \param alignment the alignment of the address
/// \param capturedByInit true if the variable is a __block variable
///   whose address is potentially changed by the initializer
void CodeGenFunction::EmitExprAsInit(const Expr *init, const ValueDecl *D,
                                     LValue lvalue, bool capturedByInit) {
  QualType type = D->getType();

  if (type->isReferenceType()) {
    RValue rvalue = EmitReferenceBindingToExpr(init);
    if (capturedByInit)
      drillIntoBlockVariable(*this, lvalue, cast<VarDecl>(D));
    EmitStoreThroughLValue(rvalue, lvalue, true);
    return;
  }
  switch (getEvaluationKind(type)) {
  case TEK_Scalar:
    EmitScalarInit(init, D, lvalue, capturedByInit);
    return;
  case TEK_Complex: {
    ComplexPairTy complex = EmitComplexExpr(init);
    if (capturedByInit)
      drillIntoBlockVariable(*this, lvalue, cast<VarDecl>(D));
    EmitStoreOfComplex(complex, lvalue, /*init*/ true);
    return;
  }
  case TEK_Aggregate:
    if (type->isAtomicType()) {
      EmitAtomicInit(const_cast<Expr*>(init), lvalue);
    } else {
      // TODO: how can we delay here if D is captured by its initializer?
      EmitAggExpr(init, AggValueSlot::forLValue(lvalue,
                                              AggValueSlot::IsDestructed,
                                         AggValueSlot::DoesNotNeedGCBarriers,
                                              AggValueSlot::IsNotAliased));
    }
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

/// Enter a destroy cleanup for the given local variable.
void CodeGenFunction::emitAutoVarTypeCleanup(
                            const CodeGenFunction::AutoVarEmission &emission,
                            QualType::DestructionKind dtorKind) {
  assert(dtorKind != QualType::DK_none);

  // Note that for __block variables, we want to destroy the
  // original stack object, not the possibly forwarded object.
  Address addr = emission.getObjectAddress(*this);

  const VarDecl *var = emission.Variable;
  QualType type = var->getType();

  CleanupKind cleanupKind = NormalAndEHCleanup;
  CodeGenFunction::Destroyer *destroyer = nullptr;

  switch (dtorKind) {
  case QualType::DK_none:
    llvm_unreachable("no cleanup for trivially-destructible variable");

  case QualType::DK_cxx_destructor:
    // If there's an NRVO flag on the emission, we need a different
    // cleanup.
    if (emission.NRVOFlag) {
      assert(!type->isArrayType());
      CXXDestructorDecl *dtor = type->getAsCXXRecordDecl()->getDestructor();
      EHStack.pushCleanup<DestroyNRVOVariable>(cleanupKind, addr,
                                               dtor, emission.NRVOFlag);
      return;
    }
    break;

  case QualType::DK_objc_strong_lifetime:
    // Suppress cleanups for pseudo-strong variables.
    if (var->isARCPseudoStrong()) return;

    // Otherwise, consider whether to use an EH cleanup or not.
    cleanupKind = getARCCleanupKind();

    // Use the imprecise destroyer by default.
    if (!var->hasAttr<ObjCPreciseLifetimeAttr>())
      destroyer = CodeGenFunction::destroyARCStrongImprecise;
    break;

  case QualType::DK_objc_weak_lifetime:
    break;
  }

  // If we haven't chosen a more specific destroyer, use the default.
  if (!destroyer) destroyer = getDestroyer(dtorKind);

  // Use an EH cleanup in array destructors iff the destructor itself
  // is being pushed as an EH cleanup.
  bool useEHCleanup = (cleanupKind & EHCleanup);
  EHStack.pushCleanup<DestroyObject>(cleanupKind, addr, type, destroyer,
                                     useEHCleanup);
}

void CodeGenFunction::EmitAutoVarCleanups(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal()) return;

  // If we don't have an insertion point, we're done.  Sema prevents
  // us from jumping into any of these scopes anyway.
  if (!HaveInsertPoint()) return;

  const VarDecl &D = *emission.Variable;

  // Make sure we call @llvm.lifetime.end.  This needs to happen
  // *last*, so the cleanup needs to be pushed *first*.
  if (emission.useLifetimeMarkers()) {
    EHStack.pushCleanup<CallLifetimeEnd>(NormalCleanup,
                                         emission.getAllocatedAddress(),
                                         emission.getSizeForLifetimeMarkers());
    EHCleanupScope &cleanup = cast<EHCleanupScope>(*EHStack.begin());
    cleanup.setLifetimeMarker();
  }

  // Check the type for a cleanup.
  if (QualType::DestructionKind dtorKind = D.getType().isDestructedType())
    emitAutoVarTypeCleanup(emission, dtorKind);

  // In GC mode, honor objc_precise_lifetime.
  if (getLangOpts().getGC() != LangOptions::NonGC &&
      D.hasAttr<ObjCPreciseLifetimeAttr>()) {
    EHStack.pushCleanup<ExtendGCLifetime>(NormalCleanup, &D);
  }

  // Handle the cleanup attribute.
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>()) {
    const FunctionDecl *FD = CA->getFunctionDecl();

    llvm::Constant *F = CGM.GetAddrOfFunction(FD);
    assert(F && "Could not find function!");

    const CGFunctionInfo &Info = CGM.getTypes().arrangeFunctionDeclaration(FD);
    EHStack.pushCleanup<CallCleanupFunction>(NormalAndEHCleanup, F, &Info, &D);
  }

  // If this is a block variable, call _Block_object_destroy
  // (on the unforwarded address).
  if (emission.IsByRef)
    enterByrefCleanup(emission);
}

CodeGenFunction::Destroyer *
CodeGenFunction::getDestroyer(QualType::DestructionKind kind) {
  switch (kind) {
  case QualType::DK_none: llvm_unreachable("no destroyer for trivial dtor");
  case QualType::DK_cxx_destructor:
    return destroyCXXObject;
  case QualType::DK_objc_strong_lifetime:
    return destroyARCStrongPrecise;
  case QualType::DK_objc_weak_lifetime:
    return destroyARCWeak;
  }
  llvm_unreachable("Unknown DestructionKind");
}

/// pushEHDestroy - Push the standard destructor for the given type as
/// an EH-only cleanup.
void CodeGenFunction::pushEHDestroy(QualType::DestructionKind dtorKind,
                                    Address addr, QualType type) {
  assert(dtorKind && "cannot push destructor for trivial type");
  assert(needsEHCleanup(dtorKind));

  pushDestroy(EHCleanup, addr, type, getDestroyer(dtorKind), true);
}

/// pushDestroy - Push the standard destructor for the given type as
/// at least a normal cleanup.
void CodeGenFunction::pushDestroy(QualType::DestructionKind dtorKind,
                                  Address addr, QualType type) {
  assert(dtorKind && "cannot push destructor for trivial type");

  CleanupKind cleanupKind = getCleanupKind(dtorKind);
  pushDestroy(cleanupKind, addr, type, getDestroyer(dtorKind),
              cleanupKind & EHCleanup);
}

void CodeGenFunction::pushDestroy(CleanupKind cleanupKind, Address addr,
                                  QualType type, Destroyer *destroyer,
                                  bool useEHCleanupForArray) {
  pushFullExprCleanup<DestroyObject>(cleanupKind, addr, type,
                                     destroyer, useEHCleanupForArray);
}

void CodeGenFunction::pushStackRestore(CleanupKind Kind, Address SPMem) {
  EHStack.pushCleanup<CallStackRestore>(Kind, SPMem);
}

void CodeGenFunction::pushLifetimeExtendedDestroy(
    CleanupKind cleanupKind, Address addr, QualType type,
    Destroyer *destroyer, bool useEHCleanupForArray) {
  assert(!isInConditionalBranch() &&
         "performing lifetime extension from within conditional");

  // Push an EH-only cleanup for the object now.
  // FIXME: When popping normal cleanups, we need to keep this EH cleanup
  // around in case a temporary's destructor throws an exception.
  if (cleanupKind & EHCleanup)
    EHStack.pushCleanup<DestroyObject>(
        static_cast<CleanupKind>(cleanupKind & ~NormalCleanup), addr, type,
        destroyer, useEHCleanupForArray);

  // Remember that we need to push a full cleanup for the object at the
  // end of the full-expression.
  pushCleanupAfterFullExpr<DestroyObject>(
      cleanupKind, addr, type, destroyer, useEHCleanupForArray);
}

/// emitDestroy - Immediately perform the destruction of the given
/// object.
///
/// \param addr - the address of the object; a type*
/// \param type - the type of the object; if an array type, all
///   objects are destroyed in reverse order
/// \param destroyer - the function to call to destroy individual
///   elements
/// \param useEHCleanupForArray - whether an EH cleanup should be
///   used when destroying array elements, in case one of the
///   destructions throws an exception
void CodeGenFunction::emitDestroy(Address addr, QualType type,
                                  Destroyer *destroyer,
                                  bool useEHCleanupForArray) {
  const ArrayType *arrayType = getContext().getAsArrayType(type);
  if (!arrayType)
    return destroyer(*this, addr, type);

  llvm::Value *length = emitArrayLength(arrayType, type, addr);

  CharUnits elementAlign =
    addr.getAlignment()
        .alignmentOfArrayElement(getContext().getTypeSizeInChars(type));

  // Normally we have to check whether the array is zero-length.
  bool checkZeroLength = true;

  // But if the array length is constant, we can suppress that.
  if (llvm::ConstantInt *constLength = dyn_cast<llvm::ConstantInt>(length)) {
    // ...and if it's constant zero, we can just skip the entire thing.
    if (constLength->isZero()) return;
    checkZeroLength = false;
  }

  llvm::Value *begin = addr.getPointer();
  llvm::Value *end = Builder.CreateInBoundsGEP(begin, length);
  emitArrayDestroy(begin, end, type, elementAlign, destroyer,
                   checkZeroLength, useEHCleanupForArray);
}

/// emitArrayDestroy - Destroys all the elements of the given array,
/// beginning from last to first.  The array cannot be zero-length.
///
/// \param begin - a type* denoting the first element of the array
/// \param end - a type* denoting one past the end of the array
/// \param elementType - the element type of the array
/// \param destroyer - the function to call to destroy elements
/// \param useEHCleanup - whether to push an EH cleanup to destroy
///   the remaining elements in case the destruction of a single
///   element throws
void CodeGenFunction::emitArrayDestroy(llvm::Value *begin,
                                       llvm::Value *end,
                                       QualType elementType,
                                       CharUnits elementAlign,
                                       Destroyer *destroyer,
                                       bool checkZeroLength,
                                       bool useEHCleanup) {
  assert(!elementType->isArrayType());

  // The basic structure here is a do-while loop, because we don't
  // need to check for the zero-element case.
  llvm::BasicBlock *bodyBB = createBasicBlock("arraydestroy.body");
  llvm::BasicBlock *doneBB = createBasicBlock("arraydestroy.done");

  if (checkZeroLength) {
    llvm::Value *isEmpty = Builder.CreateICmpEQ(begin, end,
                                                "arraydestroy.isempty");
    Builder.CreateCondBr(isEmpty, doneBB, bodyBB);
  }

  // Enter the loop body, making that address the current address.
  llvm::BasicBlock *entryBB = Builder.GetInsertBlock();
  EmitBlock(bodyBB);
  llvm::PHINode *elementPast =
    Builder.CreatePHI(begin->getType(), 2, "arraydestroy.elementPast");
  elementPast->addIncoming(end, entryBB);

  // Shift the address back by one element.
  llvm::Value *negativeOne = llvm::ConstantInt::get(SizeTy, -1, true);
  llvm::Value *element = Builder.CreateInBoundsGEP(elementPast, negativeOne,
                                                   "arraydestroy.element");

  if (useEHCleanup)
    pushRegularPartialArrayCleanup(begin, element, elementType, elementAlign,
                                   destroyer);

  // Perform the actual destruction there.
  destroyer(*this, Address(element, elementAlign), elementType);

  if (useEHCleanup)
    PopCleanupBlock();

  // Check whether we've reached the end.
  llvm::Value *done = Builder.CreateICmpEQ(element, begin, "arraydestroy.done");
  Builder.CreateCondBr(done, doneBB, bodyBB);
  elementPast->addIncoming(element, Builder.GetInsertBlock());

  // Done.
  EmitBlock(doneBB);
}

/// Perform partial array destruction as if in an EH cleanup.  Unlike
/// emitArrayDestroy, the element type here may still be an array type.
static void emitPartialArrayDestroy(CodeGenFunction &CGF,
                                    llvm::Value *begin, llvm::Value *end,
                                    QualType type, CharUnits elementAlign,
                                    CodeGenFunction::Destroyer *destroyer) {
  // If the element type is itself an array, drill down.
  unsigned arrayDepth = 0;
  while (const ArrayType *arrayType = CGF.getContext().getAsArrayType(type)) {
    // VLAs don't require a GEP index to walk into.
    if (!isa<VariableArrayType>(arrayType))
      arrayDepth++;
    type = arrayType->getElementType();
  }

  if (arrayDepth) {
    llvm::Value *zero = llvm::ConstantInt::get(CGF.SizeTy, 0);

    SmallVector<llvm::Value*,4> gepIndices(arrayDepth+1, zero);
    begin = CGF.Builder.CreateInBoundsGEP(begin, gepIndices, "pad.arraybegin");
    end = CGF.Builder.CreateInBoundsGEP(end, gepIndices, "pad.arrayend");
  }

  // Destroy the array.  We don't ever need an EH cleanup because we
  // assume that we're in an EH cleanup ourselves, so a throwing
  // destructor causes an immediate terminate.
  CGF.emitArrayDestroy(begin, end, type, elementAlign, destroyer,
                       /*checkZeroLength*/ true, /*useEHCleanup*/ false);
}

namespace {
  /// RegularPartialArrayDestroy - a cleanup which performs a partial
  /// array destroy where the end pointer is regularly determined and
  /// does not need to be loaded from a local.
  class RegularPartialArrayDestroy final : public EHScopeStack::Cleanup {
    llvm::Value *ArrayBegin;
    llvm::Value *ArrayEnd;
    QualType ElementType;
    CodeGenFunction::Destroyer *Destroyer;
    CharUnits ElementAlign;
  public:
    RegularPartialArrayDestroy(llvm::Value *arrayBegin, llvm::Value *arrayEnd,
                               QualType elementType, CharUnits elementAlign,
                               CodeGenFunction::Destroyer *destroyer)
      : ArrayBegin(arrayBegin), ArrayEnd(arrayEnd),
        ElementType(elementType), Destroyer(destroyer),
        ElementAlign(elementAlign) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      emitPartialArrayDestroy(CGF, ArrayBegin, ArrayEnd,
                              ElementType, ElementAlign, Destroyer);
    }
  };

  /// IrregularPartialArrayDestroy - a cleanup which performs a
  /// partial array destroy where the end pointer is irregularly
  /// determined and must be loaded from a local.
  class IrregularPartialArrayDestroy final : public EHScopeStack::Cleanup {
    llvm::Value *ArrayBegin;
    Address ArrayEndPointer;
    QualType ElementType;
    CodeGenFunction::Destroyer *Destroyer;
    CharUnits ElementAlign;
  public:
    IrregularPartialArrayDestroy(llvm::Value *arrayBegin,
                                 Address arrayEndPointer,
                                 QualType elementType,
                                 CharUnits elementAlign,
                                 CodeGenFunction::Destroyer *destroyer)
      : ArrayBegin(arrayBegin), ArrayEndPointer(arrayEndPointer),
        ElementType(elementType), Destroyer(destroyer),
        ElementAlign(elementAlign) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      llvm::Value *arrayEnd = CGF.Builder.CreateLoad(ArrayEndPointer);
      emitPartialArrayDestroy(CGF, ArrayBegin, arrayEnd,
                              ElementType, ElementAlign, Destroyer);
    }
  };
}

/// pushIrregularPartialArrayCleanup - Push an EH cleanup to destroy
/// already-constructed elements of the given array.  The cleanup
/// may be popped with DeactivateCleanupBlock or PopCleanupBlock.
///
/// \param elementType - the immediate element type of the array;
///   possibly still an array type
void CodeGenFunction::pushIrregularPartialArrayCleanup(llvm::Value *arrayBegin,
                                                       Address arrayEndPointer,
                                                       QualType elementType,
                                                       CharUnits elementAlign,
                                                       Destroyer *destroyer) {
  pushFullExprCleanup<IrregularPartialArrayDestroy>(EHCleanup,
                                                    arrayBegin, arrayEndPointer,
                                                    elementType, elementAlign,
                                                    destroyer);
}

/// pushRegularPartialArrayCleanup - Push an EH cleanup to destroy
/// already-constructed elements of the given array.  The cleanup
/// may be popped with DeactivateCleanupBlock or PopCleanupBlock.
///
/// \param elementType - the immediate element type of the array;
///   possibly still an array type
void CodeGenFunction::pushRegularPartialArrayCleanup(llvm::Value *arrayBegin,
                                                     llvm::Value *arrayEnd,
                                                     QualType elementType,
                                                     CharUnits elementAlign,
                                                     Destroyer *destroyer) {
  pushFullExprCleanup<RegularPartialArrayDestroy>(EHCleanup,
                                                  arrayBegin, arrayEnd,
                                                  elementType, elementAlign,
                                                  destroyer);
}

/// Lazily declare the @llvm.lifetime.start intrinsic.
llvm::Constant *CodeGenModule::getLLVMLifetimeStartFn() {
  if (LifetimeStartFn) return LifetimeStartFn;
  LifetimeStartFn = llvm::Intrinsic::getDeclaration(&getModule(),
                                            llvm::Intrinsic::lifetime_start);
  return LifetimeStartFn;
}

/// Lazily declare the @llvm.lifetime.end intrinsic.
llvm::Constant *CodeGenModule::getLLVMLifetimeEndFn() {
  if (LifetimeEndFn) return LifetimeEndFn;
  LifetimeEndFn = llvm::Intrinsic::getDeclaration(&getModule(),
                                              llvm::Intrinsic::lifetime_end);
  return LifetimeEndFn;
}

namespace {
  /// A cleanup to perform a release of an object at the end of a
  /// function.  This is used to balance out the incoming +1 of a
  /// ns_consumed argument when we can't reasonably do that just by
  /// not doing the initial retain for a __block argument.
  struct ConsumeARCParameter final : EHScopeStack::Cleanup {
    ConsumeARCParameter(llvm::Value *param,
                        ARCPreciseLifetime_t precise)
      : Param(param), Precise(precise) {}

    llvm::Value *Param;
    ARCPreciseLifetime_t Precise;

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      CGF.EmitARCRelease(Param, Precise);
    }
  };
}

/// Emit an alloca (or GlobalValue depending on target)
/// for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const VarDecl &D, ParamValue Arg,
                                   unsigned ArgNo) {
  // FIXME: Why isn't ImplicitParamDecl a ParmVarDecl?
  assert((isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D)) &&
         "Invalid argument to EmitParmDecl");

  Arg.getAnyValue()->setName(D.getName());

  QualType Ty = D.getType();

  // Use better IR generation for certain implicit parameters.
  if (auto IPD = dyn_cast<ImplicitParamDecl>(&D)) {
    // The only implicit argument a block has is its literal.
    // We assume this is always passed directly.
    if (BlockInfo) {
      setBlockContextParameter(IPD, ArgNo, Arg.getDirectValue());
      return;
    }
  }

  Address DeclPtr = Address::invalid();
  bool DoStore = false;
  bool IsScalar = hasScalarEvaluationKind(Ty);
  // If we already have a pointer to the argument, reuse the input pointer.
  if (Arg.isIndirect()) {
    DeclPtr = Arg.getIndirectAddress();
    // If we have a prettier pointer type at this point, bitcast to that.
    unsigned AS = DeclPtr.getType()->getAddressSpace();
    llvm::Type *IRTy = ConvertTypeForMem(Ty)->getPointerTo(AS);
    if (DeclPtr.getType() != IRTy)
      DeclPtr = Builder.CreateBitCast(DeclPtr, IRTy, D.getName());

    // Push a destructor cleanup for this parameter if the ABI requires it.
    // Don't push a cleanup in a thunk for a method that will also emit a
    // cleanup.
    if (!IsScalar && !CurFuncIsThunk &&
        getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
      const CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
      if (RD && RD->hasNonTrivialDestructor())
        pushDestroy(QualType::DK_cxx_destructor, DeclPtr, Ty);
    }
  } else {
    // Otherwise, create a temporary to hold the value.
    DeclPtr = CreateMemTemp(Ty, getContext().getDeclAlign(&D),
                            D.getName() + ".addr");
    DoStore = true;
  }

  llvm::Value *ArgVal = (DoStore ? Arg.getDirectValue() : nullptr);

  LValue lv = MakeAddrLValue(DeclPtr, Ty);
  if (IsScalar) {
    Qualifiers qs = Ty.getQualifiers();
    if (Qualifiers::ObjCLifetime lt = qs.getObjCLifetime()) {
      // We honor __attribute__((ns_consumed)) for types with lifetime.
      // For __strong, it's handled by just skipping the initial retain;
      // otherwise we have to balance out the initial +1 with an extra
      // cleanup to do the release at the end of the function.
      bool isConsumed = D.hasAttr<NSConsumedAttr>();

      // 'self' is always formally __strong, but if this is not an
      // init method then we don't want to retain it.
      if (D.isARCPseudoStrong()) {
        const ObjCMethodDecl *method = cast<ObjCMethodDecl>(CurCodeDecl);
        assert(&D == method->getSelfDecl());
        assert(lt == Qualifiers::OCL_Strong);
        assert(qs.hasConst());
        assert(method->getMethodFamily() != OMF_init);
        (void) method;
        lt = Qualifiers::OCL_ExplicitNone;
      }

      if (lt == Qualifiers::OCL_Strong) {
        if (!isConsumed) {
          if (CGM.getCodeGenOpts().OptimizationLevel == 0) {
            // use objc_storeStrong(&dest, value) for retaining the
            // object. But first, store a null into 'dest' because
            // objc_storeStrong attempts to release its old value.
            llvm::Value *Null = CGM.EmitNullConstant(D.getType());
            EmitStoreOfScalar(Null, lv, /* isInitialization */ true);
            EmitARCStoreStrongCall(lv.getAddress(), ArgVal, true);
            DoStore = false;
          }
          else
          // Don't use objc_retainBlock for block pointers, because we
          // don't want to Block_copy something just because we got it
          // as a parameter.
            ArgVal = EmitARCRetainNonBlock(ArgVal);
        }
      } else {
        // Push the cleanup for a consumed parameter.
        if (isConsumed) {
          ARCPreciseLifetime_t precise = (D.hasAttr<ObjCPreciseLifetimeAttr>()
                                ? ARCPreciseLifetime : ARCImpreciseLifetime);
          EHStack.pushCleanup<ConsumeARCParameter>(getARCCleanupKind(), ArgVal,
                                                   precise);
        }

        if (lt == Qualifiers::OCL_Weak) {
          EmitARCInitWeak(DeclPtr, ArgVal);
          DoStore = false; // The weak init is a store, no need to do two.
        }
      }

      // Enter the cleanup scope.
      EmitAutoVarWithLifetime(*this, D, DeclPtr, lt);
    }
  }

  // Store the initial value into the alloca.
  if (DoStore)
    EmitStoreOfScalar(ArgVal, lv, /* isInitialization */ true);

  setAddrOfLocalVar(&D, DeclPtr);

  // Emit debug info for param declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    if (CGM.getCodeGenOpts().getDebugInfo()
          >= CodeGenOptions::LimitedDebugInfo) {
      DI->EmitDeclareOfArgVariable(&D, DeclPtr.getPointer(), ArgNo, Builder);
    }
  }

  if (D.hasAttr<AnnotateAttr>())
    EmitVarAnnotations(&D, DeclPtr.getPointer());
}
