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
CIRGenFunction::emitAutoVarAlloca(const VarDecl &d) {
  QualType ty = d.getType();
  if (ty.getAddressSpace() != LangAS::Default)
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarAlloca: address space");

  mlir::Location loc = getLoc(d.getSourceRange());

  CIRGenFunction::AutoVarEmission emission(d);
  emission.IsEscapingByRef = d.isEscapingByref();
  if (emission.IsEscapingByRef)
    cgm.errorNYI(d.getSourceRange(),
                 "emitAutoVarDecl: decl escaping by reference");

  CharUnits alignment = getContext().getDeclAlign(&d);

  // If the type is variably-modified, emit all the VLA sizes for it.
  if (ty->isVariablyModifiedType())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarDecl: variably modified type");

  Address address = Address::invalid();
  if (!ty->isConstantSizeType())
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarDecl: non-constant size type");

  // A normal fixed sized variable becomes an alloca in the entry block,
  mlir::Type allocaTy = convertTypeForMem(ty);
  // Create the temp alloca and declare variable using it.
  address = createTempAlloca(allocaTy, alignment, loc, d.getName(),
                             /*insertIntoFnEntryBlock=*/false);
  declare(address.getPointer(), &d, ty, getLoc(d.getSourceRange()), alignment);

  emission.Addr = address;
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
  assert(emission.Variable && "emission was not valid!");

  // If this was emitted as a global constant, we're done.
  if (emission.wasEmittedAsGlobal())
    return;

  const VarDecl &d = *emission.Variable;

  QualType type = d.getType();

  // If this local has an initializer, emit it now.
  const Expr *init = d.getInit();

  if (!type.isPODType(getContext())) {
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarInit: non-POD type");
    return;
  }

  const Address addr = emission.Addr;

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
  if (emission.IsConstantAggregate ||
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
    // In case lv has uses it means we indeed initialized something
    // out of it while trying to build the expression, mark it as such.
    mlir::Value val = lv.getAddress().getPointer();
    assert(val && "Should have an address");
    auto allocaOp = dyn_cast_or_null<cir::AllocaOp>(val.getDefiningOp());
    assert(allocaOp && "Address should come straight out of the alloca");

    if (!allocaOp.use_empty())
      allocaOp.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
    return;
  }

  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  auto typedConstant = mlir::dyn_cast<mlir::TypedAttr>(constant);
  assert(typedConstant && "expected typed attribute");
  if (!emission.IsConstantAggregate) {
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
  const VarDecl &d = *emission.Variable;

  // Check the type for a cleanup.
  if (d.needsDestruction(getContext()))
    cgm.errorNYI(d.getSourceRange(), "emitAutoVarCleanups: type cleanup");

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

  if (d.getStorageDuration() != SD_Automatic)
    cgm.errorNYI(d.getSourceRange(), "emitVarDecl automatic storage duration");
  if (d.getType().getAddressSpace() == LangAS::opencl_local)
    cgm.errorNYI(d.getSourceRange(), "emitVarDecl openCL address space");

  assert(d.hasLocalStorage());

  CIRGenFunction::VarDeclContext varDeclCtx{*this, &d};
  return emitAutoVarDecl(d);
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
    cgm.errorNYI(init->getSourceRange(), "emitExprAsInit: reference type");
    return;
  }
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case cir::TEK_Scalar:
    emitScalarInit(init, getLoc(d->getSourceRange()), lvalue);
    return;
  case cir::TEK_Complex: {
    cgm.errorNYI(init->getSourceRange(), "emitExprAsInit: complex type");
    return;
  }
  case cir::TEK_Aggregate:
    emitAggExpr(init, AggValueSlot::forLValue(lvalue));
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void CIRGenFunction::emitDecl(const Decl &d) {
  switch (d.getKind()) {
  case Decl::Var: {
    const VarDecl &vd = cast<VarDecl>(d);
    assert(vd.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    emitVarDecl(vd);
    return;
  }
  case Decl::OpenACCDeclare:
    emitOpenACCDeclare(cast<OpenACCDeclareDecl>(d));
    return;
  case Decl::OpenACCRoutine:
    emitOpenACCRoutine(cast<OpenACCRoutineDecl>(d));
    return;
  default:
    cgm.errorNYI(d.getSourceRange(), "emitDecl: unhandled decl type");
  }
}

void CIRGenFunction::emitNullabilityCheck(LValue lhs, mlir::Value rhs,
                                          SourceLocation loc) {
  if (!sanOpts.has(SanitizerKind::NullabilityAssign))
    return;

  assert(!cir::MissingFeatures::sanitizers());
}
