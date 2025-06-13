//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delegation optimization, i.e. emitting the
/// complete constructor as a simple call to the base constructor.
bool CIRGenFunction::isConstructorDelegationValid(
    const CXXConstructorDecl *ctor) {
  // Currently we disable the optimization for classes with virtual bases
  // because (1) the address of parameter variables need to be consistent across
  // all initializers but (2) the delegate function call necessarily creates a
  // second copy of the parameter variable.
  //
  // The limiting example (purely theoretical AFAIK):
  //   struct A { A(int &c) { c++; } };
  //   struct A : virtual A {
  //     B(int count) : A(count) { printf("%d\n", count); }
  //   };
  // ...although even this example could in principle be emitted as a delegation
  // since the address of the parameter doesn't escape.
  if (ctor->getParent()->getNumVBases())
    return false;

  // We also disable the optimization for variadic functions because it's
  // impossible to "re-pass" varargs.
  if (ctor->getType()->castAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (ctor->isDelegatingConstructor())
    return false;

  return true;
}

/// This routine generates necessary code to initialize base classes and
/// non-static data members belonging to this constructor.
void CIRGenFunction::emitCtorPrologue(const CXXConstructorDecl *cd,
                                      CXXCtorType ctorType,
                                      FunctionArgList &args) {
  if (cd->isDelegatingConstructor())
    return emitDelegatingCXXConstructorCall(cd, args);

  if (cd->getNumCtorInitializers() != 0) {
    // There's much more to do here.
    cgm.errorNYI(cd->getSourceRange(), "emitCtorPrologue: any initializer");
    return;
  }
}

Address CIRGenFunction::loadCXXThisAddress() {
  assert(curFuncDecl && "loading 'this' without a func declaration?");
  assert(isa<CXXMethodDecl>(curFuncDecl));

  // Lazily compute CXXThisAlignment.
  if (cxxThisAlignment.isZero()) {
    // Just use the best known alignment for the parent.
    // TODO: if we're currently emitting a complete-object ctor/dtor, we can
    // always use the complete-object alignment.
    auto rd = cast<CXXMethodDecl>(curFuncDecl)->getParent();
    cxxThisAlignment = cgm.getClassPointerAlignment(rd);
  }

  return Address(loadCXXThis(), cxxThisAlignment);
}

void CIRGenFunction::emitDelegateCXXConstructorCall(
    const CXXConstructorDecl *ctor, CXXCtorType ctorType,
    const FunctionArgList &args, SourceLocation loc) {
  CallArgList delegateArgs;

  FunctionArgList::const_iterator i = args.begin(), e = args.end();
  assert(i != e && "no parameters to constructor");

  // this
  Address thisAddr = loadCXXThisAddress();
  delegateArgs.add(RValue::get(thisAddr.getPointer()), (*i)->getType());
  ++i;

  // FIXME: The location of the VTT parameter in the parameter list is specific
  // to the Itanium ABI and shouldn't be hardcoded here.
  if (cgm.getCXXABI().needsVTTParameter(curGD)) {
    cgm.errorNYI(loc, "emitDelegateCXXConstructorCall: VTT parameter");
    return;
  }

  // Explicit arguments.
  for (; i != e; ++i) {
    const VarDecl *param = *i;
    // FIXME: per-argument source location
    emitDelegateCallArg(delegateArgs, param, loc);
  }

  assert(!cir::MissingFeatures::sanitizers());

  emitCXXConstructorCall(ctor, ctorType, /*ForVirtualBase=*/false,
                         /*Delegating=*/true, thisAddr, delegateArgs, loc);
}

void CIRGenFunction::emitDelegatingCXXConstructorCall(
    const CXXConstructorDecl *ctor, const FunctionArgList &args) {
  assert(ctor->isDelegatingConstructor());

  Address thisPtr = loadCXXThisAddress();

  assert(!cir::MissingFeatures::objCGC());
  assert(!cir::MissingFeatures::sanitizers());
  AggValueSlot aggSlot = AggValueSlot::forAddr(
      thisPtr, Qualifiers(), AggValueSlot::IsDestructed,
      AggValueSlot::IsNotAliased, AggValueSlot::MayOverlap,
      AggValueSlot::IsNotZeroed);

  emitAggExpr(ctor->init_begin()[0]->getInit(), aggSlot);

  const CXXRecordDecl *classDecl = ctor->getParent();
  if (cgm.getLangOpts().Exceptions && !classDecl->hasTrivialDestructor()) {
    cgm.errorNYI(ctor->getSourceRange(),
                 "emitDelegatingCXXConstructorCall: exception");
    return;
  }
}

Address CIRGenFunction::getAddressOfBaseClass(
    Address value, const CXXRecordDecl *derived,
    llvm::iterator_range<CastExpr::path_const_iterator> path,
    bool nullCheckValue, SourceLocation loc) {
  assert(!path.empty() && "Base path should not be empty!");

  if ((*path.begin())->isVirtual()) {
    // The implementation here is actually complete, but let's flag this
    // as an error until the rest of the virtual base class support is in place.
    cgm.errorNYI(loc, "getAddrOfBaseClass: virtual base");
    return Address::invalid();
  }

  // Compute the static offset of the ultimate destination within its
  // allocating subobject (the virtual base, if there is one, or else
  // the "complete" object that we see).
  CharUnits nonVirtualOffset =
      cgm.computeNonVirtualBaseClassOffset(derived, path);

  // Get the base pointer type.
  mlir::Type baseValueTy = convertType((path.end()[-1])->getType());
  assert(!cir::MissingFeatures::addressSpace());

  // The if statement here is redundant now, but it will be needed when we add
  // support for virtual base classes.
  // If there is no virtual base, use cir.base_class_addr.  It takes care of
  // the adjustment and the null pointer check.
  if (nonVirtualOffset.isZero()) {
    assert(!cir::MissingFeatures::sanitizers());
    return builder.createBaseClassAddr(getLoc(loc), value, baseValueTy, 0,
                                       /*assumeNotNull=*/true);
  }

  assert(!cir::MissingFeatures::sanitizers());

  // Apply the offset
  value = builder.createBaseClassAddr(getLoc(loc), value, baseValueTy,
                                      nonVirtualOffset.getQuantity(),
                                      /*assumeNotNull=*/true);

  // Cast to the destination type.
  value = value.withElementType(builder, baseValueTy);

  return value;
}

void CIRGenFunction::emitCXXConstructorCall(const clang::CXXConstructorDecl *d,
                                            clang::CXXCtorType type,
                                            bool forVirtualBase,
                                            bool delegating,
                                            AggValueSlot thisAVS,
                                            const clang::CXXConstructExpr *e) {
  CallArgList args;
  Address thisAddr = thisAVS.getAddress();
  QualType thisType = d->getThisType();
  mlir::Value thisPtr = thisAddr.getPointer();

  assert(!cir::MissingFeatures::addressSpace());

  args.add(RValue::get(thisPtr), thisType);

  // In LLVM Codegen: If this is a trivial constructor, just emit what's needed.
  // If this is a union copy constructor, we must emit a memcpy, because the AST
  // does not model that copy.
  assert(!cir::MissingFeatures::isMemcpyEquivalentSpecialMember());

  const FunctionProtoType *fpt = d->getType()->castAs<FunctionProtoType>();

  assert(!cir::MissingFeatures::opCallArgEvaluationOrder());

  emitCallArgs(args, fpt, e->arguments(), e->getConstructor(),
               /*ParamsToSkip=*/0);

  assert(!cir::MissingFeatures::sanitizers());
  emitCXXConstructorCall(d, type, forVirtualBase, delegating, thisAddr, args,
                         e->getExprLoc());
}

void CIRGenFunction::emitCXXConstructorCall(
    const CXXConstructorDecl *d, CXXCtorType type, bool forVirtualBase,
    bool delegating, Address thisAddr, CallArgList &args, SourceLocation loc) {

  const CXXRecordDecl *crd = d->getParent();

  // If this is a call to a trivial default constructor:
  // In LLVM: do nothing.
  // In CIR: emit as a regular call, other later passes should lower the
  // ctor call into trivial initialization.
  assert(!cir::MissingFeatures::isTrivialCtorOrDtor());

  assert(!cir::MissingFeatures::isMemcpyEquivalentSpecialMember());

  bool passPrototypeArgs = true;

  // Check whether we can actually emit the constructor before trying to do so.
  if (d->getInheritedConstructor()) {
    cgm.errorNYI(d->getSourceRange(),
                 "emitCXXConstructorCall: inherited constructor");
    return;
  }

  // Insert any ABI-specific implicit constructor arguments.
  assert(!cir::MissingFeatures::implicitConstructorArgs());

  // Emit the call.
  auto calleePtr = cgm.getAddrOfCXXStructor(GlobalDecl(d, type));
  const CIRGenFunctionInfo &info = cgm.getTypes().arrangeCXXConstructorCall(
      args, d, type, passPrototypeArgs);
  CIRGenCallee callee = CIRGenCallee::forDirect(calleePtr, GlobalDecl(d, type));
  cir::CIRCallOpInterface c;
  emitCall(info, callee, ReturnValueSlot(), args, &c, getLoc(loc));

  if (cgm.getCodeGenOpts().OptimizationLevel != 0 && !crd->isDynamicClass() &&
      type != Ctor_Base && cgm.getCodeGenOpts().StrictVTablePointers)
    cgm.errorNYI(d->getSourceRange(), "vtable assumption loads");
}
