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

static void emitLValueForAnyFieldInitialization(CIRGenFunction &cgf,
                                                CXXCtorInitializer *memberInit,
                                                LValue &lhs) {
  FieldDecl *field = memberInit->getAnyMember();
  if (memberInit->isIndirectMemberInitializer()) {
    // If we are initializing an anonymous union field, drill down to the field.
    IndirectFieldDecl *indirectField = memberInit->getIndirectMember();
    for (const auto *nd : indirectField->chain()) {
      auto *fd = cast<clang::FieldDecl>(nd);
      lhs = cgf.emitLValueForFieldInitialization(lhs, fd, fd->getName());
    }
  } else {
    lhs = cgf.emitLValueForFieldInitialization(lhs, field, field->getName());
  }
}

static void emitMemberInitializer(CIRGenFunction &cgf,
                                  const CXXRecordDecl *classDecl,
                                  CXXCtorInitializer *memberInit,
                                  const CXXConstructorDecl *constructor,
                                  FunctionArgList &args) {
  assert(memberInit->isAnyMemberInitializer() &&
         "Mush have member initializer!");
  assert(memberInit->getInit() && "Must have initializer!");

  assert(!cir::MissingFeatures::generateDebugInfo());

  // non-static data member initializers
  FieldDecl *field = memberInit->getAnyMember();
  QualType fieldType = field->getType();

  mlir::Value thisPtr = cgf.loadCXXThis();
  QualType recordTy = cgf.getContext().getTypeDeclType(classDecl);

  // If a base constructor is being emitted, create an LValue that has the
  // non-virtual alignment.
  LValue lhs = (cgf.curGD.getCtorType() == Ctor_Base)
                   ? cgf.makeNaturalAlignPointeeAddrLValue(thisPtr, recordTy)
                   : cgf.makeNaturalAlignAddrLValue(thisPtr, recordTy);

  emitLValueForAnyFieldInitialization(cgf, memberInit, lhs);

  // Special case: If we are in a copy or move constructor, and we are copying
  // an array off PODs or classes with trivial copy constructors, ignore the AST
  // and perform the copy we know is equivalent.
  // FIXME: This is hacky at best... if we had a bit more explicit information
  // in the AST, we could generalize it more easily.
  const ConstantArrayType *array =
      cgf.getContext().getAsConstantArrayType(fieldType);
  if (array && constructor->isDefaulted() &&
      constructor->isCopyOrMoveConstructor()) {
    QualType baseElementTy = cgf.getContext().getBaseElementType(array);
    // NOTE(cir): CodeGen allows record types to be memcpy'd if applicable,
    // whereas ClangIR wants to represent all object construction explicitly.
    if (!baseElementTy->isRecordType()) {
      cgf.cgm.errorNYI(memberInit->getSourceRange(),
                       "emitMemberInitializer: array of non-record type");
      return;
    }
  }

  cgf.emitInitializerForField(field, lhs, memberInit->getInit());
}

/// This routine generates necessary code to initialize base classes and
/// non-static data members belonging to this constructor.
void CIRGenFunction::emitCtorPrologue(const CXXConstructorDecl *cd,
                                      CXXCtorType ctorType,
                                      FunctionArgList &args) {
  if (cd->isDelegatingConstructor()) {
    emitDelegatingCXXConstructorCall(cd, args);
    return;
  }

  // If there are no member initializers, we can just return.
  if (cd->getNumCtorInitializers() == 0)
    return;

  const CXXRecordDecl *classDecl = cd->getParent();

  // This code doesn't use range-based iteration because we may need to emit
  // code between the virtual base initializers and the non-virtual base or
  // between the non-virtual base initializers and the member initializers.
  CXXConstructorDecl::init_const_iterator b = cd->init_begin(),
                                          e = cd->init_end();

  // Virtual base initializers first, if any. They aren't needed if:
  // - This is a base ctor variant
  // - There are no vbases
  // - The class is abstract, so a complete object of it cannot be constructed
  //
  // The check for an abstract class is necessary because sema may not have
  // marked virtual base destructors referenced.
  bool constructVBases = ctorType != Ctor_Base &&
                         classDecl->getNumVBases() != 0 &&
                         !classDecl->isAbstract();
  if (constructVBases) {
    cgm.errorNYI(cd->getSourceRange(), "emitCtorPrologue: virtual base");
    return;
  }

  if ((*b)->isBaseInitializer()) {
    cgm.errorNYI(cd->getSourceRange(),
                 "emitCtorPrologue: non-virtual base initializer");
    return;
  }

  if (classDecl->isDynamicClass()) {
    cgm.errorNYI(cd->getSourceRange(),
                 "emitCtorPrologue: initialize vtable pointers");
    return;
  }

  // Finally, initialize class members.
  FieldConstructionScope fcs(*this, loadCXXThisAddress());
  // Classic codegen uses a special class to attempt to replace member
  // initializers with memcpy. We could possibly defer that to the
  // lowering or optimization phases to keep the memory accesses more
  // explicit. For now, we don't insert memcpy at all.
  assert(!cir::MissingFeatures::ctorMemcpyizer());
  for (; b != e; b++) {
    CXXCtorInitializer *member = (*b);
    assert(!member->isBaseInitializer());
    assert(member->isAnyMemberInitializer() &&
           "Delegating initializer on non-delegating constructor");
    emitMemberInitializer(*this, cd->getParent(), member, cd, args);
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

void CIRGenFunction::emitInitializerForField(FieldDecl *field, LValue lhs,
                                             Expr *init) {
  QualType fieldType = field->getType();
  switch (getEvaluationKind(fieldType)) {
  case cir::TEK_Scalar:
    if (lhs.isSimple())
      emitExprAsInit(init, field, lhs, false);
    else
      cgm.errorNYI(field->getSourceRange(),
                   "emitInitializerForField: non-simple scalar");
    break;
  case cir::TEK_Complex:
    cgm.errorNYI(field->getSourceRange(), "emitInitializerForField: complex");
    break;
  case cir::TEK_Aggregate: {
    cgm.errorNYI(field->getSourceRange(), "emitInitializerForField: aggregate");
    break;
  }
  }

  // Ensure that we destroy this object if an exception is thrown later in the
  // constructor.
  QualType::DestructionKind dtorKind = fieldType.isDestructedType();
  (void)dtorKind;
  assert(!cir::MissingFeatures::requiresCleanups());
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
