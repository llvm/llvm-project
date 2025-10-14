//===----------------------------------------------------------------------===//
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

#include "CIRGenConstantEmitter.h"
#include "CIRGenFunction.h"
#include "mlir/IR/Location.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclOpenACC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenFunction::AutoVarEmission
CIRGenFunction::emitAutoVarAlloca(const VarDecl &d,
                                  mlir::OpBuilder::InsertPoint ip) {
  QualType ty = d.getType();
  if (ty.getAddressSpace() != LangAS::Default)
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarAlloca: address space");

  mlir::Location loc = getLoc(d.getSourceRange());
  bool nrvo =
      getContext().getLangOpts().ElideConstructors && d.isNRVOVariable();

  CIRGenFunction::AutoVarEmission emission(d);
  emission.isEscapingByRef = d.isEscapingByref();
  if (emission.isEscapingByRef)
    cgm.errorNYI(d.getSourceRange(),
                 "emitAutoVarDecl: decl escaping by reference");

  CharUnits alignment = getContext().getDeclAlign(&d);

  // If the type is variably-modified, emit all the VLA sizes for it.
  if (ty->isVariablyModifiedType())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarDecl: variably modified type");

  assert(!cir::MissingFeatures::openMP());

  Address address = Address::invalid();
  if (!ty->isConstantSizeType())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarDecl: non-constant size type");

  // A normal fixed sized variable becomes an alloca in the entry block,
  // unless:
  // - it's an NRVO variable.
  // - we are compiling OpenMP and it's an OpenMP local variable.
  if (nrvo) {
    // The named return value optimization: allocate this variable in the
    // return slot, so that we can elide the copy when returning this
    // variable (C++0x [class.copy]p34).
    address = returnValue;

    if (const RecordDecl *rd = ty->getAsRecordDecl()) {
      if (const auto *cxxrd = dyn_cast<CXXRecordDecl>(rd);
          (cxxrd && !cxxrd->hasTrivialDestructor()) ||
          rd->isNonTrivialToPrimitiveDestroy())
        cgm.errorNYI(d.getSourceRange(), "emitAutoVarAlloca: set NRVO flag");
    }
  } else {
    // A normal fixed sized variable becomes an alloca in the entry block,
    mlir::Type allocaTy = convertTypeForMem(ty);
    // Create the temp alloca and declare variable using it.
    address = createTempAlloca(allocaTy, alignment, loc, d.getName(),
                               /*arraySize=*/nullptr, /*alloca=*/nullptr, ip);
    declare(address.getPointer(), &d, ty, getLoc(d.getSourceRange()),
            alignment);
  }

  emission.addr = address;
  setAddrOfLocalVar(&d, address);

  return emission;
}

/// Determine whether the given initializer is trivial in the sense
/// that it requires no code to be generated.
bool CIRGenFunction::isTrivialInitializer(const Expr *init) {
  if (!init)
    return true;

  if (const CXXConstructExpr *construct = dyn_cast<CXXConstructExpr>(init))
    if (CXXConstructorDecl *constructor = construct->getConstructor())
      if (constructor->isTrivial() && constructor->isDefaultConstructor() &&
          !construct->requiresZeroInitialization())
        return true;

  return false;
}

void CIRGenFunction::emitAutoVarInit(
    const CIRGenFunction::AutoVarEmission &emission) {
  assert(emission.variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal())
    return;

  const VarDecl &d = *emission.variable;

  QualType type = d.getType();

  // If this local has an initializer, emit it now.
  const Expr *init = d.getInit();

  // Initialize the variable here if it doesn't have a initializer and it is a
  // C struct that is non-trivial to initialize or an array containing such a
  // struct.
  if (!init && type.isNonTrivialToPrimitiveDefaultInitialize() ==
                   QualType::PDIK_Struct) {
    cgm.errorNYI(d.getSourceRange(),
                 "emitAutoVarInit: non-trivial to default initialize");
    return;
  }

  const Address addr = emission.addr;

  // Check whether this is a byref variable that's potentially
  // captured and moved by its own initializer.  If so, we'll need to
  // emit the initializer first, then copy into the variable.
  assert(!cir::MissingFeatures::opAllocaCaptureByInit());

  // Note: constexpr already initializes everything correctly.
  LangOptions::TrivialAutoVarInitKind trivialAutoVarInit =
      (d.isConstexpr()
           ? LangOptions::TrivialAutoVarInitKind::Uninitialized
           : (d.getAttr<UninitializedAttr>()
                  ? LangOptions::TrivialAutoVarInitKind::Uninitialized
                  : getContext().getLangOpts().getTrivialAutoVarInit()));

  auto initializeWhatIsTechnicallyUninitialized = [&](Address addr) {
    if (trivialAutoVarInit ==
        LangOptions::TrivialAutoVarInitKind::Uninitialized)
      return;

    cgm.errorNYI(d.getSourceRange(), "emitAutoVarInit: trivial initialization");
  };

  if (isTrivialInitializer(init)) {
    initializeWhatIsTechnicallyUninitialized(addr);
    return;
  }

  mlir::Attribute constant;
  if (emission.isConstantAggregate ||
      d.mightBeUsableInConstantExpressions(getContext())) {
    // FIXME: Differently from LLVM we try not to emit / lower too much
    // here for CIR since we are interested in seeing the ctor in some
    // analysis later on. So CIR's implementation of ConstantEmitter will
    // frequently return an empty Attribute, to signal we want to codegen
    // some trivial ctor calls and whatnots.
    constant = ConstantEmitter(*this).tryEmitAbstractForInitializer(d);
    if (constant && !mlir::isa<cir::ZeroAttr>(constant) &&
        (trivialAutoVarInit !=
         LangOptions::TrivialAutoVarInitKind::Uninitialized)) {
      cgm.errorNYI(d.getSourceRange(), "emitAutoVarInit: constant aggregate");
      return;
    }
  }

  // NOTE(cir): In case we have a constant initializer, we can just emit a
  // store. But, in CIR, we wish to retain any ctor calls, so if it is a
  // CXX temporary object creation, we ensure the ctor call is used deferring
  // its removal/optimization to the CIR lowering.
  if (!constant || isa<CXXTemporaryObjectExpr>(init)) {
    initializeWhatIsTechnicallyUninitialized(addr);
    LValue lv = makeAddrLValue(addr, type, AlignmentSource::Decl);
    emitExprAsInit(init, &d, lv);

    if (!emission.wasEmittedAsOffloadClause()) {
      // In case lv has uses it means we indeed initialized something
      // out of it while trying to build the expression, mark it as such.
      mlir::Value val = lv.getAddress().getPointer();
      assert(val && "Should have an address");
      auto allocaOp = val.getDefiningOp<cir::AllocaOp>();
      assert(allocaOp && "Address should come straight out of the alloca");

      if (!allocaOp.use_empty())
        allocaOp.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
    }

    return;
  }

  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  auto typedConstant = mlir::dyn_cast<mlir::TypedAttr>(constant);
  assert(typedConstant && "expected typed attribute");
  if (!emission.isConstantAggregate) {
    // For simple scalar/complex initialization, store the value directly.
    LValue lv = makeAddrLValue(addr, type);
    assert(init && "expected initializer");
    mlir::Location initLoc = getLoc(init->getSourceRange());
    // lv.setNonGC(true);
    return emitStoreThroughLValue(
        RValue::get(builder.getConstant(initLoc, typedConstant)), lv);
  }
}

void CIRGenFunction::emitAutoVarCleanups(
    const CIRGenFunction::AutoVarEmission &emission) {
  const VarDecl &d = *emission.variable;

  // Check the type for a cleanup.
  if (QualType::DestructionKind dtorKind = d.needsDestruction(getContext()))
    emitAutoVarTypeCleanup(emission, dtorKind);

  assert(!cir::MissingFeatures::opAllocaPreciseLifetime());

  // Handle the cleanup attribute.
  if (d.hasAttr<CleanupAttr>())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarCleanups: CleanupAttr");
}

/// Emit code and set up symbol table for a variable declaration with auto,
/// register, or no storage class specifier. These turn into simple stack
/// objects, globals depending on target.
void CIRGenFunction::emitAutoVarDecl(const VarDecl &d) {
  CIRGenFunction::AutoVarEmission emission = emitAutoVarAlloca(d);
  emitAutoVarInit(emission);
  emitAutoVarCleanups(emission);
}

void CIRGenFunction::emitVarDecl(const VarDecl &d) {
  // If the declaration has external storage, don't emit it now, allow it to be
  // emitted lazily on its first use.
  if (d.hasExternalStorage())
    return;

  if (d.getStorageDuration() != SD_Automatic) {
    // Static sampler variables translated to function calls.
    if (d.getType()->isSamplerT()) {
      // Nothing needs to be done here, but let's flag it as an error until we
      // have a test. It requires OpenCL support.
      cgm.errorNYI(d.getSourceRange(), "emitVarDecl static sampler type");
      return;
    }

    cir::GlobalLinkageKind linkage =
        cgm.getCIRLinkageVarDefinition(&d, /*IsConstant=*/false);

    // FIXME: We need to force the emission/use of a guard variable for
    // some variables even if we can constant-evaluate them because
    // we can't guarantee every translation unit will constant-evaluate them.

    return emitStaticVarDecl(d, linkage);
  }

  if (d.getType().getAddressSpace() == LangAS::opencl_local)
    cgm.errorNYI(d.getSourceRange(), "emitVarDecl openCL address space");

  assert(d.hasLocalStorage());

  CIRGenFunction::VarDeclContext varDeclCtx{*this, &d};
  return emitAutoVarDecl(d);
}

static std::string getStaticDeclName(CIRGenModule &cgm, const VarDecl &d) {
  if (cgm.getLangOpts().CPlusPlus)
    return cgm.getMangledName(&d).str();

  // If this isn't C++, we don't need a mangled name, just a pretty one.
  assert(!d.isExternallyVisible() && "name shouldn't matter");
  std::string contextName;
  const DeclContext *dc = d.getDeclContext();
  if (auto *cd = dyn_cast<CapturedDecl>(dc))
    dc = cast<DeclContext>(cd->getNonClosureContext());
  if (const auto *fd = dyn_cast<FunctionDecl>(dc))
    contextName = std::string(cgm.getMangledName(fd));
  else if (isa<BlockDecl>(dc))
    cgm.errorNYI(d.getSourceRange(), "block decl context for static var");
  else if (isa<ObjCMethodDecl>(dc))
    cgm.errorNYI(d.getSourceRange(), "ObjC decl context for static var");
  else
    cgm.errorNYI(d.getSourceRange(), "Unknown context for static var decl");

  contextName += "." + d.getNameAsString();
  return contextName;
}

// TODO(cir): LLVM uses a Constant base class. Maybe CIR could leverage an
// interface for all constants?
cir::GlobalOp
CIRGenModule::getOrCreateStaticVarDecl(const VarDecl &d,
                                       cir::GlobalLinkageKind linkage) {
  // In general, we don't always emit static var decls once before we reference
  // them. It is possible to reference them before emitting the function that
  // contains them, and it is possible to emit the containing function multiple
  // times.
  if (cir::GlobalOp existingGV = getStaticLocalDeclAddress(&d))
    return existingGV;

  QualType ty = d.getType();
  assert(ty->isConstantSizeType() && "VLAs can't be static");

  // Use the label if the variable is renamed with the asm-label extension.
  if (d.hasAttr<AsmLabelAttr>())
    errorNYI(d.getSourceRange(), "getOrCreateStaticVarDecl: asm label");

  std::string name = getStaticDeclName(*this, d);

  mlir::Type lty = getTypes().convertTypeForMem(ty);
  assert(!cir::MissingFeatures::addressSpace());

  if (d.hasAttr<LoaderUninitializedAttr>() || d.hasAttr<CUDASharedAttr>())
    errorNYI(d.getSourceRange(),
             "getOrCreateStaticVarDecl: LoaderUninitializedAttr");
  assert(!cir::MissingFeatures::addressSpace());

  mlir::Attribute init = builder.getZeroInitAttr(convertType(ty));

  cir::GlobalOp gv = builder.createVersionedGlobal(
      getModule(), getLoc(d.getLocation()), name, lty, false, linkage);
  // TODO(cir): infer visibility from linkage in global op builder.
  gv.setVisibility(getMLIRVisibilityFromCIRLinkage(linkage));
  gv.setInitialValueAttr(init);
  gv.setAlignment(getASTContext().getDeclAlign(&d).getAsAlign().value());

  if (supportsCOMDAT() && gv.isWeakForLinker())
    gv.setComdat(true);

  assert(!cir::MissingFeatures::opGlobalThreadLocal());

  setGVProperties(gv, &d);

  // OG checks if the expected address space, denoted by the type, is the
  // same as the actual address space indicated by attributes. If they aren't
  // the same, an addrspacecast is emitted when this variable is accessed.
  // In CIR however, cir.get_global already carries that information in
  // !cir.ptr type - if this global is in OpenCL local address space, then its
  // type would be !cir.ptr<..., addrspace(offload_local)>. Therefore we don't
  // need an explicit address space cast in CIR: they will get emitted when
  // lowering to LLVM IR.

  // Ensure that the static local gets initialized by making sure the parent
  // function gets emitted eventually.
  const Decl *dc = cast<Decl>(d.getDeclContext());

  // We can't name blocks or captured statements directly, so try to emit their
  // parents.
  if (isa<BlockDecl>(dc) || isa<CapturedDecl>(dc)) {
    dc = dc->getNonClosureContext();
    // FIXME: Ensure that global blocks get emitted.
    if (!dc)
      errorNYI(d.getSourceRange(), "non-closure context");
  }

  GlobalDecl gd;
  if (isa<CXXConstructorDecl>(dc))
    errorNYI(d.getSourceRange(), "C++ constructors static var context");
  else if (isa<CXXDestructorDecl>(dc))
    errorNYI(d.getSourceRange(), "C++ destructors static var context");
  else if (const auto *fd = dyn_cast<FunctionDecl>(dc))
    gd = GlobalDecl(fd);
  else {
    // Don't do anything for Obj-C method decls or global closures. We should
    // never defer them.
    assert(isa<ObjCMethodDecl>(dc) && "unexpected parent code decl");
  }
  if (gd.getDecl() && cir::MissingFeatures::openMP()) {
    // Disable emission of the parent function for the OpenMP device codegen.
    errorNYI(d.getSourceRange(), "OpenMP");
  }

  return gv;
}

/// Add the initializer for 'd' to the global variable that has already been
/// created for it. If the initializer has a different type than gv does, this
/// may free gv and return a different one. Otherwise it just returns gv.
cir::GlobalOp CIRGenFunction::addInitializerToStaticVarDecl(
    const VarDecl &d, cir::GlobalOp gv, cir::GetGlobalOp gvAddr) {
  ConstantEmitter emitter(*this);
  mlir::TypedAttr init =
      mlir::cast<mlir::TypedAttr>(emitter.tryEmitForInitializer(d));

  // If constant emission failed, then this should be a C++ static
  // initializer.
  if (!init) {
    cgm.errorNYI(d.getSourceRange(), "static var without initializer");
    return gv;
  }

  // TODO(cir): There should be debug code here to assert that the decl size
  // matches the CIR data layout type alloc size, but the code for calculating
  // the type alloc size is not implemented yet.
  assert(!cir::MissingFeatures::dataLayoutTypeAllocSize());

  // The initializer may differ in type from the global. Rewrite
  // the global to match the initializer.  (We have to do this
  // because some types, like unions, can't be completely represented
  // in the LLVM type system.)
  if (gv.getSymType() != init.getType()) {
    gv.setSymType(init.getType());

    // Normally this should be done with a call to cgm.replaceGlobal(oldGV, gv),
    // but since at this point the current block hasn't been really attached,
    // there's no visibility into the GetGlobalOp corresponding to this Global.
    // Given those constraints, thread in the GetGlobalOp and update it
    // directly.
    assert(!cir::MissingFeatures::addressSpace());
    gvAddr.getAddr().setType(builder.getPointerTo(init.getType()));
  }

  bool needsDtor =
      d.needsDestruction(getContext()) == QualType::DK_cxx_destructor;

  assert(!cir::MissingFeatures::opGlobalConstant());
  gv.setInitialValueAttr(init);

  emitter.finalize(gv);

  if (needsDtor) {
    // We have a constant initializer, but a nontrivial destructor. We still
    // need to perform a guarded "initialization" in order to register the
    // destructor.
    cgm.errorNYI(d.getSourceRange(), "C++ guarded init");
  }

  return gv;
}

void CIRGenFunction::emitStaticVarDecl(const VarDecl &d,
                                       cir::GlobalLinkageKind linkage) {
  // Check to see if we already have a global variable for this
  // declaration.  This can happen when double-emitting function
  // bodies, e.g. with complete and base constructors.
  cir::GlobalOp globalOp = cgm.getOrCreateStaticVarDecl(d, linkage);
  // TODO(cir): we should have a way to represent global ops as values without
  // having to emit a get global op. Sometimes these emissions are not used.
  mlir::Value addr = builder.createGetGlobal(globalOp);
  auto getAddrOp = addr.getDefiningOp<cir::GetGlobalOp>();
  assert(getAddrOp && "expected cir::GetGlobalOp");

  CharUnits alignment = getContext().getDeclAlign(&d);

  // Store into LocalDeclMap before generating initializer to handle
  // circular references.
  mlir::Type elemTy = convertTypeForMem(d.getType());
  setAddrOfLocalVar(&d, Address(addr, elemTy, alignment));

  // We can't have a VLA here, but we can have a pointer to a VLA,
  // even though that doesn't really make any sense.
  // Make sure to evaluate VLA bounds now so that we have them for later.
  if (d.getType()->isVariablyModifiedType()) {
    cgm.errorNYI(d.getSourceRange(),
                 "emitStaticVarDecl: variably modified type");
  }

  // Save the type in case adding the initializer forces a type change.
  mlir::Type expectedType = addr.getType();

  cir::GlobalOp var = globalOp;

  assert(!cir::MissingFeatures::cudaSupport());

  // If this value has an initializer, emit it.
  if (d.getInit())
    var = addInitializerToStaticVarDecl(d, var, getAddrOp);

  var.setAlignment(alignment.getAsAlign().value());

  // There are a lot of attributes that need to be handled here. Until
  // we start to support them, we just report an error if there are any.
  if (d.hasAttrs())
    cgm.errorNYI(d.getSourceRange(), "static var with attrs");

  if (cgm.getCodeGenOpts().KeepPersistentStorageVariables)
    cgm.errorNYI(d.getSourceRange(), "static var keep persistent storage");

  // From traditional codegen:
  // We may have to cast the constant because of the initializer
  // mismatch above.
  //
  // FIXME: It is really dangerous to store this in the map; if anyone
  // RAUW's the GV uses of this constant will be invalid.
  mlir::Value castedAddr =
      builder.createBitcast(getAddrOp.getAddr(), expectedType);
  localDeclMap.find(&d)->second = Address(castedAddr, elemTy, alignment);
  cgm.setStaticLocalDeclAddress(&d, var);

  assert(!cir::MissingFeatures::sanitizers());
  assert(!cir::MissingFeatures::generateDebugInfo());
}

void CIRGenFunction::emitScalarInit(const Expr *init, mlir::Location loc,
                                    LValue lvalue, bool capturedByInit) {
  assert(!cir::MissingFeatures::objCLifetime());

  SourceLocRAIIObject locRAII{*this, loc};
  mlir::Value value = emitScalarExpr(init);
  if (capturedByInit) {
    cgm.errorNYI(init->getSourceRange(), "emitScalarInit: captured by init");
    return;
  }
  assert(!cir::MissingFeatures::emitNullabilityCheck());
  emitStoreThroughLValue(RValue::get(value), lvalue, true);
}

void CIRGenFunction::emitExprAsInit(const Expr *init, const ValueDecl *d,
                                    LValue lvalue, bool capturedByInit) {
  SourceLocRAIIObject loc{*this, getLoc(init->getSourceRange())};
  if (capturedByInit) {
    cgm.errorNYI(init->getSourceRange(), "emitExprAsInit: captured by init");
    return;
  }

  QualType type = d->getType();

  if (type->isReferenceType()) {
    RValue rvalue = emitReferenceBindingToExpr(init);
    if (capturedByInit)
      cgm.errorNYI(init->getSourceRange(), "emitExprAsInit: captured by init");
    emitStoreThroughLValue(rvalue, lvalue);
    return;
  }
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case cir::TEK_Scalar:
    emitScalarInit(init, getLoc(d->getSourceRange()), lvalue);
    return;
  case cir::TEK_Complex: {
    mlir::Value complex = emitComplexExpr(init);
    if (capturedByInit)
      cgm.errorNYI(init->getSourceRange(),
                   "emitExprAsInit: complex type captured by init");
    mlir::Location loc = getLoc(init->getExprLoc());
    emitStoreOfComplex(loc, complex, lvalue,
                       /*isInit*/ true);
    return;
  }
  case cir::TEK_Aggregate:
    // The overlap flag here should be calculated.
    assert(!cir::MissingFeatures::aggValueSlotMayOverlap());
    emitAggExpr(init,
                AggValueSlot::forLValue(lvalue, AggValueSlot::IsDestructed,
                                        AggValueSlot::IsNotAliased,
                                        AggValueSlot::MayOverlap));
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void CIRGenFunction::emitDecl(const Decl &d, bool evaluateConditionDecl) {
  switch (d.getKind()) {
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
  case Decl::OutlinedFunction:
  case Decl::Captured:
  case Decl::UsingShadow:
  case Decl::ConstructorUsingShadow:
  case Decl::ObjCTypeParam:
  case Decl::Binding:
  case Decl::UnresolvedUsingIfExists:
  case Decl::HLSLBuffer:
  case Decl::HLSLRootSignature:
    llvm_unreachable("Declaration should not be in declstmts!");

  case Decl::Function:     // void X();
  case Decl::EnumConstant: // enum ? { X = ? }
  case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
  case Decl::Label:        // __label__ x;
  case Decl::Import:
  case Decl::MSGuid: // __declspec(uuid("..."))
  case Decl::TemplateParamObject:
  case Decl::OMPThreadPrivate:
  case Decl::OMPGroupPrivate:
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

  case Decl::Enum:      // enum X;
  case Decl::Record:    // struct/union/class X;
  case Decl::CXXRecord: // struct/union/class X; [C++]
  case Decl::NamespaceAlias:
  case Decl::Using:          // using X; [C++]
  case Decl::UsingEnum:      // using enum X; [C++]
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(!cir::MissingFeatures::generateDebugInfo());
    return;
  case Decl::Var:
  case Decl::Decomposition: {
    const VarDecl &vd = cast<VarDecl>(d);
    assert(vd.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    emitVarDecl(vd);
    if (evaluateConditionDecl)
      maybeEmitDeferredVarDeclInit(&vd);
    return;
  }
  case Decl::OpenACCDeclare:
    emitOpenACCDeclare(cast<OpenACCDeclareDecl>(d));
    return;
  case Decl::OpenACCRoutine:
    emitOpenACCRoutine(cast<OpenACCRoutineDecl>(d));
    return;
  case Decl::Typedef:     // typedef int X;
  case Decl::TypeAlias: { // using X = int; [C++0x]
    QualType ty = cast<TypedefNameDecl>(d).getUnderlyingType();
    assert(!cir::MissingFeatures::generateDebugInfo());
    if (ty->isVariablyModifiedType())
      cgm.errorNYI(d.getSourceRange(), "emitDecl: variably modified type");
    return;
  }
  case Decl::ImplicitConceptSpecialization:
  case Decl::TopLevelStmt:
  case Decl::UsingPack:
  case Decl::OMPDeclareMapper:
  case Decl::OMPDeclareReduction:
    cgm.errorNYI(d.getSourceRange(),
                 std::string("emitDecl: unhandled decl type: ") +
                     d.getDeclKindName());
  }
}

void CIRGenFunction::emitNullabilityCheck(LValue lhs, mlir::Value rhs,
                                          SourceLocation loc) {
  if (!sanOpts.has(SanitizerKind::NullabilityAssign))
    return;

  assert(!cir::MissingFeatures::sanitizers());
}

namespace {
struct DestroyObject final : EHScopeStack::Cleanup {
  DestroyObject(Address addr, QualType type,
                CIRGenFunction::Destroyer *destroyer)
      : addr(addr), type(type), destroyer(destroyer) {}

  Address addr;
  QualType type;
  CIRGenFunction::Destroyer *destroyer;

  void emit(CIRGenFunction &cgf) override {
    cgf.emitDestroy(addr, type, destroyer);
  }
};
} // namespace

void CIRGenFunction::pushDestroy(CleanupKind cleanupKind, Address addr,
                                 QualType type, Destroyer *destroyer) {
  pushFullExprCleanup<DestroyObject>(cleanupKind, addr, type, destroyer);
}

/// Destroys all the elements of the given array, beginning from last to first.
/// The array cannot be zero-length.
///
/// \param begin - a type* denoting the first element of the array
/// \param numElements - the number of elements in the array
/// \param elementType - the element type of the array
/// \param destroyer - the function to call to destroy elements
void CIRGenFunction::emitArrayDestroy(mlir::Value begin,
                                      mlir::Value numElements,
                                      QualType elementType,
                                      CharUnits elementAlign,
                                      Destroyer *destroyer) {
  assert(!elementType->isArrayType());

  // Differently from LLVM traditional codegen, use a higher level
  // representation instead of lowering directly to a loop.
  mlir::Type cirElementType = convertTypeForMem(elementType);
  cir::PointerType ptrToElmType = builder.getPointerTo(cirElementType);

  uint64_t size = 0;

  // Optimize for a constant array size.
  if (auto constantCount = numElements.getDefiningOp<cir::ConstantOp>()) {
    if (auto constIntAttr = constantCount.getValueAttr<cir::IntAttr>())
      size = constIntAttr.getUInt();
  } else {
    cgm.errorNYI(begin.getDefiningOp()->getLoc(),
                 "dynamic-length array expression");
  }

  auto arrayTy = cir::ArrayType::get(cirElementType, size);
  mlir::Value arrayOp = builder.createPtrBitcast(begin, arrayTy);

  // Emit the dtor call that will execute for every array element.
  cir::ArrayDtor::create(
      builder, *currSrcLoc, arrayOp,
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto arg = b.getInsertionBlock()->addArgument(ptrToElmType, loc);
        Address curAddr = Address(arg, cirElementType, elementAlign);
        assert(!cir::MissingFeatures::dtorCleanups());

        // Perform the actual destruction there.
        destroyer(*this, curAddr, elementType);

        cir::YieldOp::create(builder, loc);
      });
}

/// Immediately perform the destruction of the given object.
///
/// \param addr - the address of the object; a type*
/// \param type - the type of the object; if an array type, all
///   objects are destroyed in reverse order
/// \param destroyer - the function to call to destroy individual
///   elements
void CIRGenFunction::emitDestroy(Address addr, QualType type,
                                 Destroyer *destroyer) {
  const ArrayType *arrayType = getContext().getAsArrayType(type);
  if (!arrayType)
    return destroyer(*this, addr, type);

  mlir::Value length = emitArrayLength(arrayType, type, addr);

  CharUnits elementAlign = addr.getAlignment().alignmentOfArrayElement(
      getContext().getTypeSizeInChars(type));

  auto constantCount = length.getDefiningOp<cir::ConstantOp>();
  if (!constantCount) {
    assert(!cir::MissingFeatures::vlas());
    cgm.errorNYI("emitDestroy: variable length array");
    return;
  }

  auto constIntAttr = mlir::dyn_cast<cir::IntAttr>(constantCount.getValue());
  // If it's constant zero, we can just skip the entire thing.
  if (constIntAttr && constIntAttr.getUInt() == 0)
    return;

  mlir::Value begin = addr.getPointer();
  emitArrayDestroy(begin, length, type, elementAlign, destroyer);

  // If the array destroy didn't use the length op, we can erase it.
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
    cgm.errorNYI("getDestroyer: other destruction kind");
    return nullptr;
  }
  llvm_unreachable("Unknown DestructionKind");
}

/// Enter a destroy cleanup for the given local variable.
void CIRGenFunction::emitAutoVarTypeCleanup(
    const CIRGenFunction::AutoVarEmission &emission,
    QualType::DestructionKind dtorKind) {
  assert(dtorKind != QualType::DK_none);

  // Note that for __block variables, we want to destroy the
  // original stack object, not the possibly forwarded object.
  Address addr = emission.getObjectAddress(*this);

  const VarDecl *var = emission.variable;
  QualType type = var->getType();

  CleanupKind cleanupKind = NormalAndEHCleanup;
  CIRGenFunction::Destroyer *destroyer = nullptr;

  switch (dtorKind) {
  case QualType::DK_none:
    llvm_unreachable("no cleanup for trivially-destructible variable");

  case QualType::DK_cxx_destructor:
    // If there's an NRVO flag on the emission, we need a different
    // cleanup.
    if (emission.nrvoFlag) {
      cgm.errorNYI(var->getSourceRange(), "emitAutoVarTypeCleanup: NRVO");
      return;
    }
    // Otherwise, this is handled below.
    break;

  case QualType::DK_objc_strong_lifetime:
  case QualType::DK_objc_weak_lifetime:
  case QualType::DK_nontrivial_c_struct:
    cgm.errorNYI(var->getSourceRange(),
                 "emitAutoVarTypeCleanup: other dtor kind");
    return;
  }

  // If we haven't chosen a more specific destroyer, use the default.
  if (!destroyer)
    destroyer = getDestroyer(dtorKind);

  assert(!cir::MissingFeatures::ehCleanupFlags());
  ehStack.pushCleanup<DestroyObject>(cleanupKind, addr, type, destroyer);
}

void CIRGenFunction::maybeEmitDeferredVarDeclInit(const VarDecl *vd) {
  if (auto *dd = dyn_cast_if_present<DecompositionDecl>(vd)) {
    for (auto *b : dd->flat_bindings())
      if (auto *hd = b->getHoldingVar())
        emitVarDecl(*hd);
  }
}
