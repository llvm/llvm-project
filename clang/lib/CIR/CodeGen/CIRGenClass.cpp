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
