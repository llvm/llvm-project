//===--- CIRGenDecl.cpp - Emit CIR Code for declarations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRDataLayout.h"
#include "CIRGenBuilder.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "EHScopeStack.h"
#include "UnimplementedFeatureGuarding.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace cir;
using namespace clang;

CIRGenFunction::AutoVarEmission
CIRGenFunction::buildAutoVarAlloca(const VarDecl &D,
                                   mlir::OpBuilder::InsertPoint ip) {
  QualType Ty = D.getType();
  // TODO: (|| Ty.getAddressSpace() == LangAS::opencl_private &&
  //        getLangOpts().OpenCL))
  assert(!UnimplementedFeature::openCL());
  assert(Ty.getAddressSpace() == LangAS::Default);
  assert(!D.hasAttr<AnnotateAttr>() && "not implemented");

  auto loc = getLoc(D.getSourceRange());
  bool NRVO =
      getContext().getLangOpts().ElideConstructors && D.isNRVOVariable();
  AutoVarEmission emission(D);
  bool isEscapingByRef = D.isEscapingByref();
  emission.IsEscapingByRef = isEscapingByRef;

  CharUnits alignment = getContext().getDeclAlign(&D);

  // If the type is variably-modified, emit all the VLA sizes for it.
  if (Ty->isVariablyModifiedType())
    buildVariablyModifiedType(Ty);

  assert(!UnimplementedFeature::generateDebugInfo());
  assert(!UnimplementedFeature::cxxABI());

  Address address = Address::invalid();
  Address allocaAddr = Address::invalid();
  Address openMPLocalAddr =
      getCIRGenModule().getOpenMPRuntime().getAddressOfLocalVariable(*this, &D);
  assert(!getLangOpts().OpenMPIsTargetDevice && "NYI");
  if (getLangOpts().OpenMP && openMPLocalAddr.isValid()) {
    llvm_unreachable("NYI");
  } else if (Ty->isConstantSizeType()) {
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
      // Exception is if a variable is located in non-constant address space
      // in OpenCL.
      // TODO: deal with CGM.getCodeGenOpts().MergeAllConstants
      // TODO: perhaps we don't need this at all at CIR since this can
      // be done as part of lowering down to LLVM.
      if ((!getContext().getLangOpts().OpenCL ||
           Ty.getAddressSpace() == LangAS::opencl_constant) &&
          (!NRVO && !D.isEscapingByref() &&
           CGM.isTypeConstant(Ty, /*ExcludeCtor=*/true,
                              /*ExcludeDtor=*/false))) {
        buildStaticVarDecl(D, mlir::cir::GlobalLinkageKind::InternalLinkage);

        // Signal this condition to later callbacks.
        emission.Addr = Address::invalid();
        assert(emission.wasEmittedAsGlobal());
        return emission;
      }
      // Otherwise, tell the initialization code that we're in this case.
      emission.IsConstantAggregate = true;
    }

    // A normal fixed sized variable becomes an alloca in the entry block,
    // unless:
    // - it's an NRVO variable.
    // - we are compiling OpenMP and it's an OpenMP local variable.
    if (NRVO) {
      // The named return value optimization: allocate this variable in the
      // return slot, so that we can elide the copy when returning this
      // variable (C++0x [class.copy]p34).
      address = ReturnValue;
      allocaAddr = ReturnValue;

      if (const RecordType *RecordTy = Ty->getAs<RecordType>()) {
        const auto *RD = RecordTy->getDecl();
        const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD);
        if ((CXXRD && !CXXRD->hasTrivialDestructor()) ||
            RD->isNonTrivialToPrimitiveDestroy()) {
          // In LLVM: Create a flag that is used to indicate when the NRVO was
          // applied to this variable. Set it to zero to indicate that NRVO was
          // not applied. For now, use the same approach for CIRGen until we can
          // be sure it's worth doing something more aggressive.
          auto falseNVRO = builder.getFalse(loc);
          Address NRVOFlag = CreateTempAlloca(
              falseNVRO.getType(), CharUnits::One(), loc, "nrvo",
              /*ArraySize=*/nullptr, &allocaAddr);
          assert(builder.getInsertionBlock());
          builder.createStore(loc, falseNVRO, NRVOFlag);

          // Record the NRVO flag for this variable.
          NRVOFlags[&D] = NRVOFlag.getPointer();
          emission.NRVOFlag = NRVOFlag.getPointer();
        }
      }
    } else {
      if (isEscapingByRef)
        llvm_unreachable("NYI");

      mlir::Type allocaTy = getTypes().convertTypeForMem(Ty);
      CharUnits allocaAlignment = alignment;
      // Create the temp alloca and declare variable using it.
      mlir::Value addrVal;
      address = CreateTempAlloca(allocaTy, allocaAlignment, loc, D.getName(),
                                 /*ArraySize=*/nullptr, &allocaAddr, ip);
      if (failed(declare(address, &D, Ty, getLoc(D.getSourceRange()), alignment,
                         addrVal))) {
        CGM.emitError("Cannot declare variable");
        return emission;
      }
      // TODO: what about emitting lifetime markers for MSVC catch parameters?
      // TODO: something like @llvm.lifetime.start/end here? revisit this later.
      assert(!UnimplementedFeature::shouldEmitLifetimeMarkers());
    }
  } else { // not openmp nor constant sized type
    bool VarAllocated = false;
    if (getLangOpts().OpenMPIsTargetDevice)
      llvm_unreachable("NYI");

    if (!VarAllocated) {
      if (!DidCallStackSave) {
        // Save the stack.
        auto defaultTy = AllocaInt8PtrTy;
        CharUnits Align = CharUnits::fromQuantity(
            CGM.getDataLayout().getAlignment(defaultTy, false));
        Address Stack = CreateTempAlloca(defaultTy, Align, loc, "saved_stack");

        mlir::Value V = builder.createStackSave(loc, defaultTy);
        assert(V.getType() == AllocaInt8PtrTy);
        builder.createStore(loc, V, Stack);

        DidCallStackSave = true;

        // Push a cleanup block and restore the stack there.
        // FIXME: in general circumstances, this should be an EH cleanup.
        pushStackRestore(NormalCleanup, Stack);
      }

      auto VlaSize = getVLASize(Ty);
      mlir::Type mTy = convertTypeForMem(VlaSize.Type);

      // Allocate memory for the array.
      address = CreateTempAlloca(mTy, alignment, loc, "vla", VlaSize.NumElts,
                                 &allocaAddr, builder.saveInsertionPoint());
    }

    // If we have debug info enabled, properly describe the VLA dimensions for
    // this type by registering the vla size expression for each of the
    // dimensions.
    assert(!UnimplementedFeature::generateDebugInfo());
  }

  emission.Addr = address;
  setAddrOfLocalVar(&D, emission.Addr);
  return emission;
}

/// Determine whether the given initializer is trivial in the sense
/// that it requires no code to be generated.
bool CIRGenFunction::isTrivialInitializer(const Expr *Init) {
  if (!Init)
    return true;

  if (const CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init))
    if (CXXConstructorDecl *Constructor = Construct->getConstructor())
      if (Constructor->isTrivial() && Constructor->isDefaultConstructor() &&
          !Construct->requiresZeroInitialization())
        return true;

  return false;
}

static void emitStoresForConstant(CIRGenModule &CGM, const VarDecl &D,
                                  Address addr, bool isVolatile,
                                  CIRGenBuilderTy &builder,
                                  mlir::TypedAttr constant, bool IsAutoInit) {
  auto Ty = constant.getType();
  cir::CIRDataLayout layout{CGM.getModule()};
  uint64_t ConstantSize = layout.getTypeAllocSize(Ty);
  if (!ConstantSize)
    return;
  assert(!UnimplementedFeature::addAutoInitAnnotation());
  assert(!UnimplementedFeature::vectorConstants());
  assert(!UnimplementedFeature::shouldUseBZeroPlusStoresToInitialize());
  assert(!UnimplementedFeature::shouldUseMemSetToInitialize());
  assert(!UnimplementedFeature::shouldSplitConstantStore());
  assert(!UnimplementedFeature::shouldCreateMemCpyFromGlobal());
  // In CIR we want to emit a store for the whole thing, later lowering
  // prepare to LLVM should unwrap this into the best policy (see asserts
  // above).
  //
  // FIXME(cir): This is closer to memcpy behavior but less optimal, instead of
  // copy from a global, we just create a cir.const out of it.

  if (addr.getElementType() != Ty) {
    auto ptr = addr.getPointer();
    ptr = builder.createBitcast(ptr.getLoc(), ptr, builder.getPointerTo(Ty));
    addr = addr.withPointer(ptr, addr.isKnownNonNull());
  }

  auto loc = CGM.getLoc(D.getSourceRange());
  builder.createStore(loc, builder.getConstant(loc, constant), addr);
}

void CIRGenFunction::buildAutoVarInit(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal())
    return;

  const VarDecl &D = *emission.Variable;
  QualType type = D.getType();

  // If this local has an initializer, emit it now.
  const Expr *Init = D.getInit();

  // TODO: in LLVM codegen if we are at an unreachable point, the initializer
  // isn't emitted unless it contains a label. What we want for CIR?
  assert(builder.getInsertionBlock());

  // Initialize the variable here if it doesn't have a initializer and it is a
  // C struct that is non-trivial to initialize or an array containing such a
  // struct.
  if (!Init && type.isNonTrivialToPrimitiveDefaultInitialize() ==
                   QualType::PDIK_Struct) {
    assert(0 && "not implemented");
    return;
  }

  const Address Loc = emission.Addr;
  // Check whether this is a byref variable that's potentially
  // captured and moved by its own initializer.  If so, we'll need to
  // emit the initializer first, then copy into the variable.
  assert(!UnimplementedFeature::capturedByInit() && "NYI");

  // Note: constexpr already initializes everything correctly.
  LangOptions::TrivialAutoVarInitKind trivialAutoVarInit =
      (D.isConstexpr()
           ? LangOptions::TrivialAutoVarInitKind::Uninitialized
           : (D.getAttr<UninitializedAttr>()
                  ? LangOptions::TrivialAutoVarInitKind::Uninitialized
                  : getContext().getLangOpts().getTrivialAutoVarInit()));

  auto initializeWhatIsTechnicallyUninitialized = [&](Address Loc) {
    if (trivialAutoVarInit ==
        LangOptions::TrivialAutoVarInitKind::Uninitialized)
      return;

    assert(0 && "unimplemented");
  };

  if (isTrivialInitializer(Init))
    return initializeWhatIsTechnicallyUninitialized(Loc);

  mlir::Attribute constant;
  if (emission.IsConstantAggregate ||
      D.mightBeUsableInConstantExpressions(getContext())) {
    // FIXME: Differently from LLVM we try not to emit / lower too much
    // here for CIR since we are interesting in seeing the ctor in some
    // analysis later on. So CIR's implementation of ConstantEmitter will
    // frequently return an empty Attribute, to signal we want to codegen
    // some trivial ctor calls and whatnots.
    constant = ConstantEmitter(*this).tryEmitAbstractForInitializer(D);
    if (constant && !constant.isa<mlir::cir::ZeroAttr>() &&
        (trivialAutoVarInit !=
         LangOptions::TrivialAutoVarInitKind::Uninitialized)) {
      llvm_unreachable("NYI");
    }
  }

  // NOTE(cir): In case we have a constant initializer, we can just emit a
  // store. But, in CIR, we wish to retain any ctor calls, so if it is a
  // CXX temporary object creation, we ensure the ctor call is used deferring
  // its removal/optimization to the CIR lowering.
  if (!constant || isa<CXXTemporaryObjectExpr>(Init)) {
    initializeWhatIsTechnicallyUninitialized(Loc);
    LValue lv = LValue::makeAddr(Loc, type, AlignmentSource::Decl);
    buildExprAsInit(Init, &D, lv);
    // In case lv has uses it means we indeed initialized something
    // out of it while trying to build the expression, mark it as such.
    auto addr = lv.getAddress().getPointer();
    assert(addr && "Should have an address");
    auto allocaOp = dyn_cast_or_null<mlir::cir::AllocaOp>(addr.getDefiningOp());
    assert(allocaOp && "Address should come straight out of the alloca");

    if (!allocaOp.use_empty())
      allocaOp.setInitAttr(mlir::UnitAttr::get(builder.getContext()));
    return;
  }

  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  auto typedConstant = constant.dyn_cast<mlir::TypedAttr>();
  assert(typedConstant && "expected typed attribute");
  if (!emission.IsConstantAggregate) {
    // For simple scalar/complex initialization, store the value directly.
    LValue lv = makeAddrLValue(Loc, type);
    assert(Init && "expected initializer");
    auto initLoc = getLoc(Init->getSourceRange());
    lv.setNonGC(true);
    return buildStoreThroughLValue(
        RValue::get(builder.getConstant(initLoc, typedConstant)), lv);
  }

  emitStoresForConstant(CGM, D, Loc, type.isVolatileQualified(), builder,
                        typedConstant, /*IsAutoInit=*/false);
}

void CIRGenFunction::buildAutoVarCleanups(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal())
    return;

  // TODO: in LLVM codegen if we are at an unreachable point codgen
  // is ignored. What we want for CIR?
  assert(builder.getInsertionBlock());
  const VarDecl &D = *emission.Variable;

  // Check the type for a cleanup.
  if (QualType::DestructionKind dtorKind = D.needsDestruction(getContext()))
    buildAutoVarTypeCleanup(emission, dtorKind);

  // In GC mode, honor objc_precise_lifetime.
  if (getContext().getLangOpts().getGC() != LangOptions::NonGC &&
      D.hasAttr<ObjCPreciseLifetimeAttr>())
    assert(0 && "not implemented");

  // Handle the cleanup attribute.
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>())
    assert(0 && "not implemented");

  // TODO: handle block variable
}

/// Emit code and set up symbol table for a variable declaration with auto,
/// register, or no storage class specifier. These turn into simple stack
/// objects, globals depending on target.
void CIRGenFunction::buildAutoVarDecl(const VarDecl &D) {
  AutoVarEmission emission = buildAutoVarAlloca(D);
  buildAutoVarInit(emission);
  buildAutoVarCleanups(emission);
}

void CIRGenFunction::buildVarDecl(const VarDecl &D) {
  if (D.hasExternalStorage()) {
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  }

  // Some function-scope variable does not have static storage but still
  // needs to be emitted like a static variable, e.g. a function-scope
  // variable in constant address space in OpenCL.
  if (D.getStorageDuration() != SD_Automatic) {
    // Static sampler variables translated to function calls.
    if (D.getType()->isSamplerT())
      return;

    auto Linkage = CGM.getCIRLinkageVarDefinition(&D, /*IsConstant=*/false);

    // FIXME: We need to force the emission/use of a guard variable for
    // some variables even if we can constant-evaluate them because
    // we can't guarantee every translation unit will constant-evaluate them.

    return buildStaticVarDecl(D, Linkage);
  }

  if (D.getType().getAddressSpace() == LangAS::opencl_local)
    llvm_unreachable("OpenCL and address space are NYI");

  assert(D.hasLocalStorage());

  CIRGenFunction::VarDeclContext varDeclCtx{*this, &D};
  return buildAutoVarDecl(D);
}

static std::string getStaticDeclName(CIRGenModule &CGM, const VarDecl &D) {
  if (CGM.getLangOpts().CPlusPlus)
    return CGM.getMangledName(&D).str();

  // If this isn't C++, we don't need a mangled name, just a pretty one.
  assert(!D.isExternallyVisible() && "name shouldn't matter");
  std::string ContextName;
  const DeclContext *DC = D.getDeclContext();
  if (auto *CD = dyn_cast<CapturedDecl>(DC))
    DC = cast<DeclContext>(CD->getNonClosureContext());
  if (const auto *FD = dyn_cast<FunctionDecl>(DC))
    ContextName = std::string(CGM.getMangledName(FD));
  else if (const auto *BD = dyn_cast<BlockDecl>(DC))
    llvm_unreachable("block decl context for static var is NYI");
  else if (const auto *OMD = dyn_cast<ObjCMethodDecl>(DC))
    llvm_unreachable("ObjC decl context for static var is NYI");
  else
    llvm_unreachable("Unknown context for static var decl");

  ContextName += "." + D.getNameAsString();
  return ContextName;
}

// TODO(cir): LLVM uses a Constant base class. Maybe CIR could leverage an
// interface for all constants?
mlir::cir::GlobalOp
CIRGenModule::getOrCreateStaticVarDecl(const VarDecl &D,
                                       mlir::cir::GlobalLinkageKind Linkage) {
  // In general, we don't always emit static var decls once before we reference
  // them. It is possible to reference them before emitting the function that
  // contains them, and it is possible to emit the containing function multiple
  // times.
  if (mlir::cir::GlobalOp ExistingGV = StaticLocalDeclMap[&D])
    return ExistingGV;

  QualType Ty = D.getType();
  assert(Ty->isConstantSizeType() && "VLAs can't be static");

  // Use the label if the variable is renamed with the asm-label extension.
  std::string Name;
  if (D.hasAttr<AsmLabelAttr>())
    llvm_unreachable("asm label is NYI");
  else
    Name = getStaticDeclName(*this, D);

  mlir::Type LTy = getTypes().convertTypeForMem(Ty);
  assert(!UnimplementedFeature::addressSpace());

  // OpenCL variables in local address space and CUDA shared
  // variables cannot have an initializer.
  mlir::Attribute Init = nullptr;
  if (Ty.getAddressSpace() == LangAS::opencl_local ||
      D.hasAttr<CUDASharedAttr>() || D.hasAttr<LoaderUninitializedAttr>())
    llvm_unreachable("OpenCL & CUDA are NYI");
  else
    Init = builder.getZeroInitAttr(getTypes().ConvertType(Ty));

  mlir::cir::GlobalOp GV = builder.createVersionedGlobal(
      getModule(), getLoc(D.getLocation()), Name, LTy, false, Linkage);
  // TODO(cir): infer visibility from linkage in global op builder.
  GV.setVisibility(getMLIRVisibilityFromCIRLinkage(Linkage));
  GV.setInitialValueAttr(Init);
  GV.setAlignment(getASTContext().getDeclAlign(&D).getAsAlign().value());

  if (supportsCOMDAT() && GV.isWeakForLinker())
    llvm_unreachable("COMDAT globals are NYI");

  if (D.getTLSKind())
    llvm_unreachable("TLS mode is NYI");

  setGVProperties(GV, &D);

  // Make sure the result is of the correct type.
  assert(!UnimplementedFeature::addressSpace());

  // Ensure that the static local gets initialized by making sure the parent
  // function gets emitted eventually.
  const Decl *DC = cast<Decl>(D.getDeclContext());

  // We can't name blocks or captured statements directly, so try to emit their
  // parents.
  if (isa<BlockDecl>(DC) || isa<CapturedDecl>(DC)) {
    DC = DC->getNonClosureContext();
    // FIXME: Ensure that global blocks get emitted.
    if (!DC)
      llvm_unreachable("address space is NYI");
  }

  GlobalDecl GD;
  if (const auto *CD = dyn_cast<CXXConstructorDecl>(DC))
    llvm_unreachable("C++ constructors static var context is NYI");
  else if (const auto *DD = dyn_cast<CXXDestructorDecl>(DC))
    llvm_unreachable("C++ destructors static var context is NYI");
  else if (const auto *FD = dyn_cast<FunctionDecl>(DC))
    GD = GlobalDecl(FD);
  else {
    // Don't do anything for Obj-C method decls or global closures. We should
    // never defer them.
    assert(isa<ObjCMethodDecl>(DC) && "unexpected parent code decl");
  }
  if (GD.getDecl() && UnimplementedFeature::openMP()) {
    // Disable emission of the parent function for the OpenMP device codegen.
    llvm_unreachable("OpenMP is NYI");
  }

  return GV;
}

/// Add the initializer for 'D' to the global variable that has already been
/// created for it. If the initializer has a different type than GV does, this
/// may free GV and return a different one. Otherwise it just returns GV.
mlir::cir::GlobalOp
CIRGenFunction::addInitializerToStaticVarDecl(const VarDecl &D,
                                              mlir::cir::GlobalOp GV) {
  ConstantEmitter emitter(*this);
  mlir::TypedAttr Init =
      emitter.tryEmitForInitializer(D).dyn_cast<mlir::TypedAttr>();
  assert(Init && "Expected typed attribute");

  // If constant emission failed, then this should be a C++ static
  // initializer.
  if (!Init) {
    if (!getLangOpts().CPlusPlus)
      CGM.ErrorUnsupported(D.getInit(), "constant l-value expression");
    else if (D.hasFlexibleArrayInit(getContext()))
      CGM.ErrorUnsupported(D.getInit(), "flexible array initializer");
    else {
      // Since we have a static initializer, this global variable can't
      // be constant.
      GV.setConstant(false);
      llvm_unreachable("C++ guarded init it NYI");
    }
    return GV;
  }

#ifndef NDEBUG
  CharUnits VarSize = CGM.getASTContext().getTypeSizeInChars(D.getType()) +
                      D.getFlexibleArrayInitChars(getContext());
  CharUnits CstSize = CharUnits::fromQuantity(
      CGM.getDataLayout().getTypeAllocSize(Init.getType()));
  assert(VarSize == CstSize && "Emitted constant has unexpected size");
#endif

  // The initializer may differ in type from the global. Rewrite
  // the global to match the initializer.  (We have to do this
  // because some types, like unions, can't be completely represented
  // in the LLVM type system.)
  if (GV.getSymType() != Init.getType()) {
    llvm_unreachable("static decl initializer type mismatch is NYI");
  }

  bool NeedsDtor =
      D.needsDestruction(getContext()) == QualType::DK_cxx_destructor;

  GV.setConstant(
      CGM.isTypeConstant(D.getType(), /*ExcludeCtor=*/true, !NeedsDtor));
  GV.setInitialValueAttr(Init);

  emitter.finalize(GV);

  if (NeedsDtor) {
    // We have a constant initializer, but a nontrivial destructor. We still
    // need to perform a guarded "initialization" in order to register the
    // destructor.
    llvm_unreachable("C++ guarded init is NYI");
  }

  return GV;
}

void CIRGenFunction::buildStaticVarDecl(const VarDecl &D,
                                        mlir::cir::GlobalLinkageKind Linkage) {
  // Check to see if we already have a global variable for this
  // declaration.  This can happen when double-emitting function
  // bodies, e.g. with complete and base constructors.
  auto globalOp = CGM.getOrCreateStaticVarDecl(D, Linkage);
  // TODO(cir): we should have a way to represent global ops as values without
  // having to emit a get global op. Sometimes these emissions are not used.
  auto addr = getBuilder().createGetGlobal(globalOp);
  CharUnits alignment = getContext().getDeclAlign(&D);

  // Store into LocalDeclMap before generating initializer to handle
  // circular references.
  mlir::Type elemTy = getTypes().convertTypeForMem(D.getType());
  setAddrOfLocalVar(&D, Address(addr, elemTy, alignment));

  // We can't have a VLA here, but we can have a pointer to a VLA,
  // even though that doesn't really make any sense.
  // Make sure to evaluate VLA bounds now so that we have them for later.
  if (D.getType()->isVariablyModifiedType())
    llvm_unreachable("VLAs are NYI");

  // Save the type in case adding the initializer forces a type change.
  mlir::Type expectedType = addr.getType();

  auto var = globalOp;

  // CUDA's local and local static __shared__ variables should not
  // have any non-empty initializers. This is ensured by Sema.
  // Whatever initializer such variable may have when it gets here is
  // a no-op and should not be emitted.
  bool isCudaSharedVar = getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
                         D.hasAttr<CUDASharedAttr>();
  // If this value has an initializer, emit it.
  if (D.getInit() && !isCudaSharedVar)
    var = addInitializerToStaticVarDecl(D, var);

  var.setAlignment(alignment.getAsAlign().value());

  if (D.hasAttr<AnnotateAttr>())
    llvm_unreachable("Global annotations are NYI");

  if (auto *SA = D.getAttr<PragmaClangBSSSectionAttr>())
    llvm_unreachable("CIR global BSS section attribute is NYI");
  if (auto *SA = D.getAttr<PragmaClangDataSectionAttr>())
    llvm_unreachable("CIR global Data section attribute is NYI");
  if (auto *SA = D.getAttr<PragmaClangRodataSectionAttr>())
    llvm_unreachable("CIR global Rodata section attribute is NYI");
  if (auto *SA = D.getAttr<PragmaClangRelroSectionAttr>())
    llvm_unreachable("CIR global Relro section attribute is NYI");

  if (const SectionAttr *SA = D.getAttr<SectionAttr>())
    llvm_unreachable("CIR global object file section attribute is NYI");

  if (D.hasAttr<RetainAttr>())
    llvm_unreachable("llvm.used metadata is NYI");
  else if (D.hasAttr<UsedAttr>())
    llvm_unreachable("llvm.compiler.used metadata is NYI");

  // We may have to cast the constant because of the initializer
  // mismatch above.
  //
  // FIXME: It is really dangerous to store this in the map; if anyone
  // RAUW's the GV uses of this constant will be invalid.
  // TODO(cir): its suppose to be possible that the initializer does not match
  // the static var type. When this happens, there should be a cast here.
  assert(var.getSymType() != expectedType &&
         "static var init type mismatch is NYI");
  CGM.setStaticLocalDeclAddress(&D, var);

  assert(!UnimplementedFeature::reportGlobalToASan());

  // Emit global variable debug descriptor for static vars.
  auto *DI = getDebugInfo();
  if (DI && CGM.getCodeGenOpts().hasReducedDebugInfo()) {
    llvm_unreachable("Debug info is NYI");
  }
}

void CIRGenFunction::buildNullabilityCheck(LValue LHS, mlir::Value RHS,
                                           SourceLocation Loc) {
  if (!SanOpts.has(SanitizerKind::NullabilityAssign))
    return;

  llvm_unreachable("NYI");
}

void CIRGenFunction::buildScalarInit(const Expr *init, mlir::Location loc,
                                     LValue lvalue, bool capturedByInit) {
  // TODO: this is where a lot of ObjC lifetime stuff would be done.
  SourceLocRAIIObject Loc{*this, loc};
  mlir::Value value = buildScalarExpr(init);
  buildStoreThroughLValue(RValue::get(value), lvalue);
  return;
}

void CIRGenFunction::buildExprAsInit(const Expr *init, const ValueDecl *D,
                                     LValue lvalue, bool capturedByInit) {
  SourceLocRAIIObject Loc{*this, getLoc(init->getSourceRange())};
  if (capturedByInit)
    llvm_unreachable("NYI");

  QualType type = D->getType();

  if (type->isReferenceType()) {
    RValue rvalue = buildReferenceBindingToExpr(init);
    if (capturedByInit)
      llvm_unreachable("NYI");
    buildStoreThroughLValue(rvalue, lvalue);
    return;
  }
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case TEK_Scalar:
    buildScalarInit(init, getLoc(D->getSourceRange()), lvalue);
    return;
  case TEK_Complex: {
    assert(0 && "not implemented");
    return;
  }
  case TEK_Aggregate:
    assert(!type->isAtomicType() && "NYI");
    AggValueSlot::Overlap_t Overlap = AggValueSlot::MayOverlap;
    if (isa<VarDecl>(D))
      Overlap = AggValueSlot::DoesNotOverlap;
    else if (auto *FD = dyn_cast<FieldDecl>(D))
      assert(false && "Field decl NYI");
    else
      assert(false && "Only VarDecl implemented so far");
    // TODO: how can we delay here if D is captured by its initializer?
    buildAggExpr(init,
                 AggValueSlot::forLValue(lvalue, AggValueSlot::IsDestructed,
                                         AggValueSlot::DoesNotNeedGCBarriers,
                                         AggValueSlot::IsNotAliased, Overlap));
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void CIRGenFunction::buildDecl(const Decl &D) {
  switch (D.getKind()) {
  case Decl::ImplicitConceptSpecialization:
  case Decl::HLSLBuffer:
  case Decl::TopLevelStmt:
    llvm_unreachable("NYI");
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
  case Decl::CXXDeductionGuide:
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
  case Decl::PragmaComment:
  case Decl::PragmaDetectMismatch:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::Export:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::Friend:
  case Decl::FriendTemplate:
  case Decl::Block:
  case Decl::Captured:
  case Decl::UsingShadow:
  case Decl::ConstructorUsingShadow:
  case Decl::ObjCTypeParam:
  case Decl::Binding:
  case Decl::UnresolvedUsingIfExists:
    llvm_unreachable("Declaration should not be in declstmts!");
  case Decl::Record:    // struct/union/class X;
  case Decl::CXXRecord: // struct/union/class X; [C++]
    if (auto *DI = getDebugInfo())
      llvm_unreachable("NYI");
    return;
  case Decl::Enum: // enum X;
    if (auto *DI = getDebugInfo())
      llvm_unreachable("NYI");
    return;
  case Decl::Function:     // void X();
  case Decl::EnumConstant: // enum ? { X = ? }
  case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
  case Decl::Label:        // __label__ x;
  case Decl::Import:
  case Decl::MSGuid: // __declspec(uuid("..."))
  case Decl::TemplateParamObject:
  case Decl::OMPThreadPrivate:
  case Decl::OMPAllocate:
  case Decl::OMPCapturedExpr:
  case Decl::OMPRequires:
  case Decl::Empty:
  case Decl::Concept:
  case Decl::LifetimeExtendedTemporary:
  case Decl::RequiresExprBody:
  case Decl::UnnamedGlobalConstant:
    // None of these decls require codegen support.
    return;

  case Decl::NamespaceAlias:
  case Decl::Using: // using X; [C++]
  case Decl::UsingEnum:      // using enum X; [C++]
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(!UnimplementedFeature::generateDebugInfo());
    return;
  case Decl::UsingPack:
    assert(0 && "Not implemented");
    return;
  case Decl::Var:
  case Decl::Decomposition: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    buildVarDecl(VD);
    if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
      assert(0 && "Not implemented");

    // FIXME: add this
    // if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
    //   for (auto *B : DD->bindings())
    //     if (auto *HD = B->getHoldingVar())
    //       EmitVarDecl(*HD);
    return;
  }

  case Decl::OMPDeclareReduction:
  case Decl::OMPDeclareMapper:
    assert(0 && "Not implemented");

  case Decl::Typedef:     // typedef int X;
  case Decl::TypeAlias: { // using X = int; [C++0x]
    QualType Ty = cast<TypedefNameDecl>(D).getUnderlyingType();
    if (auto *DI = getDebugInfo())
      assert(!UnimplementedFeature::generateDebugInfo());
    if (Ty->isVariablyModifiedType())
      buildVariablyModifiedType(Ty);
    return;
  }
  }
}

namespace {
struct DestroyObject final : EHScopeStack::Cleanup {
  DestroyObject(Address addr, QualType type,
                CIRGenFunction::Destroyer *destroyer, bool useEHCleanupForArray)
      : addr(addr), type(type), destroyer(destroyer),
        useEHCleanupForArray(useEHCleanupForArray) {}

  Address addr;
  QualType type;
  CIRGenFunction::Destroyer *destroyer;
  bool useEHCleanupForArray;

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    // Don't use an EH cleanup recursively from an EH cleanup.
    [[maybe_unused]] bool useEHCleanupForArray =
        flags.isForNormalCleanup() && this->useEHCleanupForArray;

    CGF.emitDestroy(addr, type, destroyer, useEHCleanupForArray);
  }
};

template <class Derived> struct DestroyNRVOVariable : EHScopeStack::Cleanup {
  DestroyNRVOVariable(Address addr, QualType type, mlir::Value NRVOFlag)
      : NRVOFlag(NRVOFlag), Loc(addr), Ty(type) {}

  mlir::Value NRVOFlag;
  Address Loc;
  QualType Ty;

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("NYI");
  }

  virtual ~DestroyNRVOVariable() = default;
};

struct DestroyNRVOVariableCXX final
    : DestroyNRVOVariable<DestroyNRVOVariableCXX> {
  DestroyNRVOVariableCXX(Address addr, QualType type,
                         const CXXDestructorDecl *Dtor, mlir::Value NRVOFlag)
      : DestroyNRVOVariable<DestroyNRVOVariableCXX>(addr, type, NRVOFlag),
        Dtor(Dtor) {}

  const CXXDestructorDecl *Dtor;

  void emitDestructorCall(CIRGenFunction &CGF) { llvm_unreachable("NYI"); }
};

struct DestroyNRVOVariableC final : DestroyNRVOVariable<DestroyNRVOVariableC> {
  DestroyNRVOVariableC(Address addr, mlir::Value NRVOFlag, QualType Ty)
      : DestroyNRVOVariable<DestroyNRVOVariableC>(addr, Ty, NRVOFlag) {}

  void emitDestructorCall(CIRGenFunction &CGF) { llvm_unreachable("NYI"); }
};

struct CallStackRestore final : EHScopeStack::Cleanup {
  Address Stack;
  CallStackRestore(Address Stack) : Stack(Stack) {}
  bool isRedundantBeforeReturn() override { return true; }
  void Emit(CIRGenFunction &CGF, Flags flags) override {
    auto loc = Stack.getPointer().getLoc();
    mlir::Value V = CGF.getBuilder().createLoad(loc, Stack);
    CGF.getBuilder().createStackRestore(loc, V);
  }
};

struct ExtendGCLifetime final : EHScopeStack::Cleanup {
  const VarDecl &Var;
  ExtendGCLifetime(const VarDecl *var) : Var(*var) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("NYI");
  }
};

struct CallCleanupFunction final : EHScopeStack::Cleanup {
  // FIXME: mlir::Value used as placeholder, check options before implementing
  // Emit below.
  mlir::Value CleanupFn;
  const CIRGenFunctionInfo &FnInfo;
  const VarDecl &Var;

  CallCleanupFunction(mlir::Value CleanupFn, const CIRGenFunctionInfo *Info,
                      const VarDecl *Var)
      : CleanupFn(CleanupFn), FnInfo(*Info), Var(*Var) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("NYI");
  }
};
} // end anonymous namespace

/// Push the standard destructor for the given type as
/// at least a normal cleanup.
void CIRGenFunction::pushDestroy(QualType::DestructionKind dtorKind,
                                 Address addr, QualType type) {
  assert(dtorKind && "cannot push destructor for trivial type");

  CleanupKind cleanupKind = getCleanupKind(dtorKind);
  pushDestroy(cleanupKind, addr, type, getDestroyer(dtorKind),
              cleanupKind & EHCleanup);
}

void CIRGenFunction::pushDestroy(CleanupKind cleanupKind, Address addr,
                                 QualType type, Destroyer *destroyer,
                                 bool useEHCleanupForArray) {
  pushFullExprCleanup<DestroyObject>(cleanupKind, addr, type, destroyer,
                                     useEHCleanupForArray);
}

namespace {
/// A cleanup which performs a partial array destroy where the end pointer is
/// regularly determined and does not need to be loaded from a local.
class RegularPartialArrayDestroy final : public EHScopeStack::Cleanup {
  mlir::Value ArrayBegin;
  mlir::Value ArrayEnd;
  QualType ElementType;
  [[maybe_unused]] CIRGenFunction::Destroyer *Destroyer;
  CharUnits ElementAlign;

public:
  RegularPartialArrayDestroy(mlir::Value arrayBegin, mlir::Value arrayEnd,
                             QualType elementType, CharUnits elementAlign,
                             CIRGenFunction::Destroyer *destroyer)
      : ArrayBegin(arrayBegin), ArrayEnd(arrayEnd), ElementType(elementType),
        Destroyer(destroyer), ElementAlign(elementAlign) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("NYI");
  }
};

/// A cleanup which performs a partial array destroy where the end pointer is
/// irregularly determined and must be loaded from a local.
class IrregularPartialArrayDestroy final : public EHScopeStack::Cleanup {
  mlir::Value ArrayBegin;
  Address ArrayEndPointer;
  QualType ElementType;
  [[maybe_unused]] CIRGenFunction::Destroyer *Destroyer;
  CharUnits ElementAlign;

public:
  IrregularPartialArrayDestroy(mlir::Value arrayBegin, Address arrayEndPointer,
                               QualType elementType, CharUnits elementAlign,
                               CIRGenFunction::Destroyer *destroyer)
      : ArrayBegin(arrayBegin), ArrayEndPointer(arrayEndPointer),
        ElementType(elementType), Destroyer(destroyer),
        ElementAlign(elementAlign) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("NYI");
  }
};
} // end anonymous namespace

/// Push an EH cleanup to destroy already-constructed elements of the given
/// array.  The cleanup may be popped with DeactivateCleanupBlock or
/// PopCleanupBlock.
///
/// \param elementType - the immediate element type of the array;
///   possibly still an array type
void CIRGenFunction::pushIrregularPartialArrayCleanup(mlir::Value arrayBegin,
                                                      Address arrayEndPointer,
                                                      QualType elementType,
                                                      CharUnits elementAlign,
                                                      Destroyer *destroyer) {
  pushFullExprCleanup<IrregularPartialArrayDestroy>(
      EHCleanup, arrayBegin, arrayEndPointer, elementType, elementAlign,
      destroyer);
}

/// Push an EH cleanup to destroy already-constructed elements of the given
/// array.  The cleanup may be popped with DeactivateCleanupBlock or
/// PopCleanupBlock.
///
/// \param elementType - the immediate element type of the array;
///   possibly still an array type
void CIRGenFunction::pushRegularPartialArrayCleanup(mlir::Value arrayBegin,
                                                    mlir::Value arrayEnd,
                                                    QualType elementType,
                                                    CharUnits elementAlign,
                                                    Destroyer *destroyer) {
  pushFullExprCleanup<RegularPartialArrayDestroy>(
      EHCleanup, arrayBegin, arrayEnd, elementType, elementAlign, destroyer);
}

/// Destroys all the elements of the given array, beginning from last to first.
/// The array cannot be zero-length.
///
/// \param begin - a type* denoting the first element of the array
/// \param end - a type* denoting one past the end of the array
/// \param elementType - the element type of the array
/// \param destroyer - the function to call to destroy elements
/// \param useEHCleanup - whether to push an EH cleanup to destroy
///   the remaining elements in case the destruction of a single
///   element throws
void CIRGenFunction::buildArrayDestroy(mlir::Value begin, mlir::Value end,
                                       QualType elementType,
                                       CharUnits elementAlign,
                                       Destroyer *destroyer,
                                       bool checkZeroLength,
                                       bool useEHCleanup) {
  assert(!elementType->isArrayType());
  if (checkZeroLength) {
    llvm_unreachable("NYI");
  }

  // Differently from LLVM traditional codegen, use a higher level
  // representation instead of lowering directly to a loop.
  mlir::Type cirElementType = convertTypeForMem(elementType);
  auto ptrToElmType = builder.getPointerTo(cirElementType);

  // Emit the dtor call that will execute for every array element.
  builder.create<mlir::cir::ArrayDtor>(
      *currSrcLoc, begin, [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto arg = b.getInsertionBlock()->addArgument(ptrToElmType, loc);
        Address curAddr = Address(arg, ptrToElmType, elementAlign);
        if (useEHCleanup) {
          pushRegularPartialArrayCleanup(arg, arg, elementType, elementAlign,
                                         destroyer);
        }

        // Perform the actual destruction there.
        destroyer(*this, curAddr, elementType);

        if (useEHCleanup)
          PopCleanupBlock();

        builder.create<mlir::cir::YieldOp>(loc);
      });
}

/// Immediately perform the destruction of the given object.
///
/// \param addr - the address of the object; a type*
/// \param type - the type of the object; if an array type, all
///   objects are destroyed in reverse order
/// \param destroyer - the function to call to destroy individual
///   elements
/// \param useEHCleanupForArray - whether an EH cleanup should be
///   used when destroying array elements, in case one of the
///   destructions throws an exception
void CIRGenFunction::emitDestroy(Address addr, QualType type,
                                 Destroyer *destroyer,
                                 bool useEHCleanupForArray) {
  const ArrayType *arrayType = getContext().getAsArrayType(type);
  if (!arrayType)
    return destroyer(*this, addr, type);

  auto length = buildArrayLength(arrayType, type, addr);

  CharUnits elementAlign = addr.getAlignment().alignmentOfArrayElement(
      getContext().getTypeSizeInChars(type));

  // Normally we have to check whether the array is zero-length.
  bool checkZeroLength = true;

  // But if the array length is constant, we can suppress that.
  auto constantCount = dyn_cast<mlir::cir::ConstantOp>(length.getDefiningOp());
  if (constantCount) {
    auto constIntAttr = constantCount.getValue().dyn_cast<mlir::cir::IntAttr>();
    // ...and if it's constant zero, we can just skip the entire thing.
    if (constIntAttr && constIntAttr.getUInt() == 0)
      return;
    checkZeroLength = false;
  } else {
    llvm_unreachable("NYI");
  }

  auto begin = addr.getPointer();
  mlir::Value end; // Use this for future non-constant counts.
  buildArrayDestroy(begin, end, type, elementAlign, destroyer, checkZeroLength,
                    useEHCleanupForArray);
  if (constantCount.use_empty())
    constantCount.erase();
}

CIRGenFunction::Destroyer *
CIRGenFunction::getDestroyer(QualType::DestructionKind kind) {
  switch (kind) {
  case QualType::DK_none:
    llvm_unreachable("no destroyer for trivial dtor");
  case QualType::DK_cxx_destructor:
    return destroyCXXObject;
  case QualType::DK_objc_strong_lifetime:
  case QualType::DK_objc_weak_lifetime:
  case QualType::DK_nontrivial_c_struct:
    llvm_unreachable("NYI");
  }
  llvm_unreachable("Unknown DestructionKind");
}

void CIRGenFunction::pushStackRestore(CleanupKind Kind, Address SPMem) {
  EHStack.pushCleanup<CallStackRestore>(Kind, SPMem);
}

/// Enter a destroy cleanup for the given local variable.
void CIRGenFunction::buildAutoVarTypeCleanup(
    const CIRGenFunction::AutoVarEmission &emission,
    QualType::DestructionKind dtorKind) {
  assert(dtorKind != QualType::DK_none);

  // Note that for __block variables, we want to destroy the
  // original stack object, not the possibly forwarded object.
  Address addr = emission.getObjectAddress(*this);

  const VarDecl *var = emission.Variable;
  QualType type = var->getType();

  CleanupKind cleanupKind = NormalAndEHCleanup;
  CIRGenFunction::Destroyer *destroyer = nullptr;

  switch (dtorKind) {
  case QualType::DK_none:
    llvm_unreachable("no cleanup for trivially-destructible variable");

  case QualType::DK_cxx_destructor:
    // If there's an NRVO flag on the emission, we need a different
    // cleanup.
    if (emission.NRVOFlag) {
      assert(!type->isArrayType());
      CXXDestructorDecl *dtor = type->getAsCXXRecordDecl()->getDestructor();
      EHStack.pushCleanup<DestroyNRVOVariableCXX>(cleanupKind, addr, type, dtor,
                                                  emission.NRVOFlag);
      return;
    }
    break;

  case QualType::DK_objc_strong_lifetime:
    llvm_unreachable("NYI");
    break;

  case QualType::DK_objc_weak_lifetime:
    break;

  case QualType::DK_nontrivial_c_struct:
    llvm_unreachable("NYI");
  }

  // If we haven't chosen a more specific destroyer, use the default.
  if (!destroyer)
    destroyer = getDestroyer(dtorKind);

  // Use an EH cleanup in array destructors iff the destructor itself
  // is being pushed as an EH cleanup.
  bool useEHCleanup = (cleanupKind & EHCleanup);
  EHStack.pushCleanup<DestroyObject>(cleanupKind, addr, type, destroyer,
                                     useEHCleanup);
}

/// Push the standard destructor for the given type as an EH-only cleanup.
void CIRGenFunction::pushEHDestroy(QualType::DestructionKind dtorKind,
                                   Address addr, QualType type) {
  assert(dtorKind && "cannot push destructor for trivial type");
  assert(needsEHCleanup(dtorKind));

  pushDestroy(EHCleanup, addr, type, getDestroyer(dtorKind), true);
}
