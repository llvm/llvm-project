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
#include "CIRGenValue.h"

#include "clang/AST/EvaluatedExprVisitor.h"
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
         "Must have member initializer!");
  assert(memberInit->getInit() && "Must have initializer!");

  assert(!cir::MissingFeatures::generateDebugInfo());

  // non-static data member initializers
  FieldDecl *field = memberInit->getAnyMember();
  QualType fieldType = field->getType();

  mlir::Value thisPtr = cgf.loadCXXThis();
  CanQualType recordTy = cgf.getContext().getCanonicalTagType(classDecl);

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

static bool isInitializerOfDynamicClass(const CXXCtorInitializer *baseInit) {
  const Type *baseType = baseInit->getBaseClass();
  const auto *baseClassDecl = baseType->castAsCXXRecordDecl();
  return baseClassDecl->isDynamicClass();
}

namespace {
/// Call the destructor for a direct base class.
struct CallBaseDtor final : EHScopeStack::Cleanup {
  const CXXRecordDecl *baseClass;
  bool baseIsVirtual;
  CallBaseDtor(const CXXRecordDecl *base, bool baseIsVirtual)
      : baseClass(base), baseIsVirtual(baseIsVirtual) {}

  void emit(CIRGenFunction &cgf) override {
    const CXXRecordDecl *derivedClass =
        cast<CXXMethodDecl>(cgf.curFuncDecl)->getParent();

    const CXXDestructorDecl *d = baseClass->getDestructor();
    // We are already inside a destructor, so presumably the object being
    // destroyed should have the expected type.
    QualType thisTy = d->getFunctionObjectParameterType();
    assert(cgf.currSrcLoc && "expected source location");
    Address addr = cgf.getAddressOfDirectBaseInCompleteClass(
        *cgf.currSrcLoc, cgf.loadCXXThisAddress(), derivedClass, baseClass,
        baseIsVirtual);
    cgf.emitCXXDestructorCall(d, Dtor_Base, baseIsVirtual,
                              /*delegating=*/false, addr, thisTy);
  }
};

/// A visitor which checks whether an initializer uses 'this' in a
/// way which requires the vtable to be properly set.
struct DynamicThisUseChecker
    : ConstEvaluatedExprVisitor<DynamicThisUseChecker> {
  using super = ConstEvaluatedExprVisitor<DynamicThisUseChecker>;

  bool usesThis = false;

  DynamicThisUseChecker(const ASTContext &c) : super(c) {}

  // Black-list all explicit and implicit references to 'this'.
  //
  // Do we need to worry about external references to 'this' derived
  // from arbitrary code? If so, then anything which runs arbitrary
  // external code might potentially access the vtable.
  void VisitCXXThisExpr(const CXXThisExpr *e) { usesThis = true; }
};
} // end anonymous namespace

static bool baseInitializerUsesThis(ASTContext &c, const Expr *init) {
  DynamicThisUseChecker checker(c);
  checker.Visit(init);
  return checker.usesThis;
}

/// Gets the address of a direct base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'thisAddr' is assumed to be non-null.
Address CIRGenFunction::getAddressOfDirectBaseInCompleteClass(
    mlir::Location loc, Address thisAddr, const CXXRecordDecl *derived,
    const CXXRecordDecl *base, bool baseIsVirtual) {
  // 'thisAddr' must be a pointer (in some address space) to Derived.
  assert(thisAddr.getElementType() == convertType(derived));

  // Compute the offset of the virtual base.
  CharUnits offset;
  const ASTRecordLayout &layout = getContext().getASTRecordLayout(derived);
  if (baseIsVirtual)
    offset = layout.getVBaseClassOffset(base);
  else
    offset = layout.getBaseClassOffset(base);

  return builder.createBaseClassAddr(loc, thisAddr, convertType(base),
                                     offset.getQuantity(),
                                     /*assumeNotNull=*/true);
}

void CIRGenFunction::emitBaseInitializer(mlir::Location loc,
                                         const CXXRecordDecl *classDecl,
                                         CXXCtorInitializer *baseInit) {
  assert(curFuncDecl && "loading 'this' without a func declaration?");
  assert(isa<CXXMethodDecl>(curFuncDecl));

  assert(baseInit->isBaseInitializer() && "Must have base initializer!");

  Address thisPtr = loadCXXThisAddress();

  const Type *baseType = baseInit->getBaseClass();
  const auto *baseClassDecl = baseType->castAsCXXRecordDecl();

  bool isBaseVirtual = baseInit->isBaseVirtual();

  // If the initializer for the base (other than the constructor
  // itself) accesses 'this' in any way, we need to initialize the
  // vtables.
  if (baseInitializerUsesThis(getContext(), baseInit->getInit()))
    initializeVTablePointers(loc, classDecl);

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  Address v = getAddressOfDirectBaseInCompleteClass(
      loc, thisPtr, classDecl, baseClassDecl, isBaseVirtual);
  assert(!cir::MissingFeatures::aggValueSlotGC());
  AggValueSlot aggSlot = AggValueSlot::forAddr(
      v, Qualifiers(), AggValueSlot::IsDestructed, AggValueSlot::IsNotAliased,
      getOverlapForBaseInit(classDecl, baseClassDecl, isBaseVirtual));

  emitAggExpr(baseInit->getInit(), aggSlot);

  assert(!cir::MissingFeatures::requiresCleanups());
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

  const CXXRecordDecl *classDecl = cd->getParent();

  // Virtual base initializers aren't needed if:
  // - This is a base ctor variant
  // - There are no vbases
  // - The class is abstract, so a complete object of it cannot be constructed
  //
  // The check for an abstract class is necessary because sema may not have
  // marked virtual base destructors referenced.
  bool constructVBases = ctorType != Ctor_Base &&
                         classDecl->getNumVBases() != 0 &&
                         !classDecl->isAbstract();
  if (constructVBases &&
      !cgm.getTarget().getCXXABI().hasConstructorVariants()) {
    cgm.errorNYI(cd->getSourceRange(),
                 "emitCtorPrologue: virtual base without variants");
    return;
  }

  // Create three separate ranges for the different types of initializers.
  auto allInits = cd->inits();

  // Find the boundaries between the three groups.
  auto virtualBaseEnd = std::find_if(
      allInits.begin(), allInits.end(), [](const CXXCtorInitializer *Init) {
        return !(Init->isBaseInitializer() && Init->isBaseVirtual());
      });

  auto nonVirtualBaseEnd = std::find_if(virtualBaseEnd, allInits.end(),
                                        [](const CXXCtorInitializer *Init) {
                                          return !Init->isBaseInitializer();
                                        });

  // Create the three ranges.
  auto virtualBaseInits = llvm::make_range(allInits.begin(), virtualBaseEnd);
  auto nonVirtualBaseInits =
      llvm::make_range(virtualBaseEnd, nonVirtualBaseEnd);
  auto memberInits = llvm::make_range(nonVirtualBaseEnd, allInits.end());

  const mlir::Value oldThisValue = cxxThisValue;

  auto emitInitializer = [&](CXXCtorInitializer *baseInit) {
    if (cgm.getCodeGenOpts().StrictVTablePointers &&
        cgm.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(baseInit)) {
      // It's OK to continue after emitting the error here. The missing code
      // just "launders" the 'this' pointer.
      cgm.errorNYI(cd->getSourceRange(),
                   "emitCtorPrologue: strict vtable pointers for vbase");
    }
    emitBaseInitializer(getLoc(cd->getBeginLoc()), classDecl, baseInit);
  };

  // Process virtual base initializers.
  for (CXXCtorInitializer *virtualBaseInit : virtualBaseInits) {
    if (!constructVBases)
      continue;
    emitInitializer(virtualBaseInit);
  }

  assert(!cir::MissingFeatures::msabi());

  // Then, non-virtual base initializers.
  for (CXXCtorInitializer *nonVirtualBaseInit : nonVirtualBaseInits) {
    assert(!nonVirtualBaseInit->isBaseVirtual());
    emitInitializer(nonVirtualBaseInit);
  }

  cxxThisValue = oldThisValue;

  initializeVTablePointers(getLoc(cd->getBeginLoc()), classDecl);

  // Finally, initialize class members.
  FieldConstructionScope fcs(*this, loadCXXThisAddress());
  // Classic codegen uses a special class to attempt to replace member
  // initializers with memcpy. We could possibly defer that to the
  // lowering or optimization phases to keep the memory accesses more
  // explicit. For now, we don't insert memcpy at all.
  assert(!cir::MissingFeatures::ctorMemcpyizer());
  for (CXXCtorInitializer *member : memberInits) {
    assert(!member->isBaseInitializer());
    assert(member->isAnyMemberInitializer() &&
           "Delegating initializer on non-delegating constructor");
    emitMemberInitializer(*this, cd->getParent(), member, cd, args);
  }
}

static Address applyNonVirtualAndVirtualOffset(
    mlir::Location loc, CIRGenFunction &cgf, Address addr,
    CharUnits nonVirtualOffset, mlir::Value virtualOffset,
    const CXXRecordDecl *derivedClass, const CXXRecordDecl *nearestVBase,
    mlir::Type baseValueTy = {}, bool assumeNotNull = true) {
  // Assert that we have something to do.
  assert(!nonVirtualOffset.isZero() || virtualOffset != nullptr);

  // Compute the offset from the static and dynamic components.
  mlir::Value baseOffset;
  if (!nonVirtualOffset.isZero()) {
    if (virtualOffset) {
      cgf.cgm.errorNYI(
          loc,
          "applyNonVirtualAndVirtualOffset: virtual and non-virtual offset");
      return Address::invalid();
    } else {
      assert(baseValueTy && "expected base type");
      // If no virtualOffset is present this is the final stop.
      return cgf.getBuilder().createBaseClassAddr(
          loc, addr, baseValueTy, nonVirtualOffset.getQuantity(),
          assumeNotNull);
    }
  } else {
    baseOffset = virtualOffset;
  }

  // Apply the base offset.  cir.ptr_stride adjusts by a number of elements,
  // not bytes.  So the pointer must be cast to a byte pointer and back.

  mlir::Value ptr = addr.getPointer();
  mlir::Type charPtrType = cgf.cgm.UInt8PtrTy;
  mlir::Value charPtr = cgf.getBuilder().createBitcast(ptr, charPtrType);
  mlir::Value adjusted = cir::PtrStrideOp::create(
      cgf.getBuilder(), loc, charPtrType, charPtr, baseOffset);
  ptr = cgf.getBuilder().createBitcast(adjusted, ptr.getType());

  // If we have a virtual component, the alignment of the result will
  // be relative only to the known alignment of that vbase.
  CharUnits alignment;
  if (virtualOffset) {
    assert(nearestVBase && "virtual offset without vbase?");
    alignment = cgf.cgm.getVBaseAlignment(addr.getAlignment(), derivedClass,
                                          nearestVBase);
  } else {
    alignment = addr.getAlignment();
  }
  alignment = alignment.alignmentAtOffset(nonVirtualOffset);

  return Address(ptr, alignment);
}

void CIRGenFunction::initializeVTablePointer(mlir::Location loc,
                                             const VPtr &vptr) {
  // Compute the address point.
  mlir::Value vtableAddressPoint =
      cgm.getCXXABI().getVTableAddressPointInStructor(
          *this, vptr.vtableClass, vptr.base, vptr.nearestVBase);

  if (!vtableAddressPoint)
    return;

  // Compute where to store the address point.
  mlir::Value virtualOffset{};
  CharUnits nonVirtualOffset = CharUnits::Zero();

  mlir::Type baseValueTy;
  if (cgm.getCXXABI().isVirtualOffsetNeededForVTableField(*this, vptr)) {
    // We need to use the virtual base offset offset because the virtual base
    // might have a different offset in the most derived class.
    virtualOffset = cgm.getCXXABI().getVirtualBaseClassOffset(
        loc, *this, loadCXXThisAddress(), vptr.vtableClass, vptr.nearestVBase);
    nonVirtualOffset = vptr.offsetFromNearestVBase;
  } else {
    // We can just use the base offset in the complete class.
    nonVirtualOffset = vptr.base.getBaseOffset();
    baseValueTy =
        convertType(getContext().getCanonicalTagType(vptr.base.getBase()));
  }

  // Apply the offsets.
  Address classAddr = loadCXXThisAddress();
  if (!nonVirtualOffset.isZero() || virtualOffset) {
    classAddr = applyNonVirtualAndVirtualOffset(
        loc, *this, classAddr, nonVirtualOffset, virtualOffset,
        vptr.vtableClass, vptr.nearestVBase, baseValueTy);
  }

  // Finally, store the address point. Use the same CIR types as the field.
  //
  // vtable field is derived from `this` pointer, therefore they should be in
  // the same addr space.
  assert(!cir::MissingFeatures::addressSpace());
  auto vtablePtr = cir::VTableGetVPtrOp::create(
      builder, loc, builder.getPtrToVPtrType(), classAddr.getPointer());
  Address vtableField = Address(vtablePtr, classAddr.getAlignment());
  builder.createStore(loc, vtableAddressPoint, vtableField);
  assert(!cir::MissingFeatures::opTBAA());
  assert(!cir::MissingFeatures::createInvariantGroup());
}

void CIRGenFunction::initializeVTablePointers(mlir::Location loc,
                                              const CXXRecordDecl *rd) {
  // Ignore classes without a vtable.
  if (!rd->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  if (cgm.getCXXABI().doStructorsInitializeVPtrs(rd))
    for (const auto &vptr : getVTablePointers(rd))
      initializeVTablePointer(loc, vptr);

  if (rd->getNumVBases())
    cgm.getCXXABI().initializeHiddenVirtualInheritanceMembers(*this, rd);
}

CIRGenFunction::VPtrsVector
CIRGenFunction::getVTablePointers(const CXXRecordDecl *vtableClass) {
  CIRGenFunction::VPtrsVector vptrsResult;
  VisitedVirtualBasesSetTy vbases;
  getVTablePointers(BaseSubobject(vtableClass, CharUnits::Zero()),
                    /*NearestVBase=*/nullptr,
                    /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                    /*BaseIsNonVirtualPrimaryBase=*/false, vtableClass, vbases,
                    vptrsResult);
  return vptrsResult;
}

void CIRGenFunction::getVTablePointers(BaseSubobject base,
                                       const CXXRecordDecl *nearestVBase,
                                       CharUnits offsetFromNearestVBase,
                                       bool baseIsNonVirtualPrimaryBase,
                                       const CXXRecordDecl *vtableClass,
                                       VisitedVirtualBasesSetTy &vbases,
                                       VPtrsVector &vptrs) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!baseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    VPtr vptr = {base, nearestVBase, offsetFromNearestVBase, vtableClass};
    vptrs.push_back(vptr);
  }

  const CXXRecordDecl *rd = base.getBase();

  for (const auto &nextBase : rd->bases()) {
    const auto *baseDecl =
        cast<CXXRecordDecl>(nextBase.getType()->castAs<RecordType>()->getDecl())
            ->getDefinitionOrSelf();

    // Ignore classes without a vtable.
    if (!baseDecl->isDynamicClass())
      continue;

    CharUnits baseOffset;
    CharUnits baseOffsetFromNearestVBase;
    bool baseDeclIsNonVirtualPrimaryBase;
    const CXXRecordDecl *nextBaseDecl;

    if (nextBase.isVirtual()) {
      // Check if we've visited this virtual base before.
      if (!vbases.insert(baseDecl).second)
        continue;

      const ASTRecordLayout &layout =
          getContext().getASTRecordLayout(vtableClass);

      nextBaseDecl = baseDecl;
      baseOffset = layout.getVBaseClassOffset(baseDecl);
      baseOffsetFromNearestVBase = CharUnits::Zero();
      baseDeclIsNonVirtualPrimaryBase = false;
    } else {
      const ASTRecordLayout &layout = getContext().getASTRecordLayout(rd);

      nextBaseDecl = nearestVBase;
      baseOffset = base.getBaseOffset() + layout.getBaseClassOffset(baseDecl);
      baseOffsetFromNearestVBase =
          offsetFromNearestVBase + layout.getBaseClassOffset(baseDecl);
      baseDeclIsNonVirtualPrimaryBase = layout.getPrimaryBase() == baseDecl;
    }

    getVTablePointers(BaseSubobject(baseDecl, baseOffset), nextBaseDecl,
                      baseOffsetFromNearestVBase,
                      baseDeclIsNonVirtualPrimaryBase, vtableClass, vbases,
                      vptrs);
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
    if (lhs.isSimple()) {
      emitExprAsInit(init, field, lhs, false);
    } else {
      RValue rhs = RValue::get(emitScalarExpr(init));
      emitStoreThroughLValue(rhs, lhs);
    }
    break;
  case cir::TEK_Complex:
    emitComplexExprIntoLValue(init, lhs, /*isInit=*/true);
    break;
  case cir::TEK_Aggregate: {
    assert(!cir::MissingFeatures::aggValueSlotGC());
    assert(!cir::MissingFeatures::sanitizers());
    AggValueSlot slot = AggValueSlot::forLValue(
        lhs, AggValueSlot::IsDestructed, AggValueSlot::IsNotAliased,
        getOverlapForFieldInit(field), AggValueSlot::IsNotZeroed);
    emitAggExpr(init, slot);
    break;
  }
  }

  // Ensure that we destroy this object if an exception is thrown later in the
  // constructor.
  QualType::DestructionKind dtorKind = fieldType.isDestructedType();
  (void)dtorKind;
  assert(!cir::MissingFeatures::requiresCleanups());
}

CharUnits
CIRGenModule::getDynamicOffsetAlignment(CharUnits actualBaseAlign,
                                        const CXXRecordDecl *baseDecl,
                                        CharUnits expectedTargetAlign) {
  // If the base is an incomplete type (which is, alas, possible with
  // member pointers), be pessimistic.
  if (!baseDecl->isCompleteDefinition())
    return std::min(actualBaseAlign, expectedTargetAlign);

  const ASTRecordLayout &baseLayout =
      getASTContext().getASTRecordLayout(baseDecl);
  CharUnits expectedBaseAlign = baseLayout.getNonVirtualAlignment();

  // If the class is properly aligned, assume the target offset is, too.
  //
  // This actually isn't necessarily the right thing to do --- if the
  // class is a complete object, but it's only properly aligned for a
  // base subobject, then the alignments of things relative to it are
  // probably off as well.  (Note that this requires the alignment of
  // the target to be greater than the NV alignment of the derived
  // class.)
  //
  // However, our approach to this kind of under-alignment can only
  // ever be best effort; after all, we're never going to propagate
  // alignments through variables or parameters.  Note, in particular,
  // that constructing a polymorphic type in an address that's less
  // than pointer-aligned will generally trap in the constructor,
  // unless we someday add some sort of attribute to change the
  // assumed alignment of 'this'.  So our goal here is pretty much
  // just to allow the user to explicitly say that a pointer is
  // under-aligned and then safely access its fields and vtables.
  if (actualBaseAlign >= expectedBaseAlign)
    return expectedTargetAlign;

  // Otherwise, we might be offset by an arbitrary multiple of the
  // actual alignment.  The correct adjustment is to take the min of
  // the two alignments.
  return std::min(actualBaseAlign, expectedTargetAlign);
}

/// Return the best known alignment for a pointer to a virtual base,
/// given the alignment of a pointer to the derived class.
clang::CharUnits
CIRGenModule::getVBaseAlignment(CharUnits actualDerivedAlign,
                                const CXXRecordDecl *derivedClass,
                                const CXXRecordDecl *vbaseClass) {
  // The basic idea here is that an underaligned derived pointer might
  // indicate an underaligned base pointer.

  assert(vbaseClass->isCompleteDefinition());
  const ASTRecordLayout &baseLayout =
      getASTContext().getASTRecordLayout(vbaseClass);
  CharUnits expectedVBaseAlign = baseLayout.getNonVirtualAlignment();

  return getDynamicOffsetAlignment(actualDerivedAlign, derivedClass,
                                   expectedVBaseAlign);
}

/// Emit a loop to call a particular constructor for each of several members
/// of an array.
///
/// \param ctor the constructor to call for each element
/// \param arrayType the type of the array to initialize
/// \param arrayBegin an arrayType*
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CIRGenFunction::emitCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, const clang::ArrayType *arrayType,
    Address arrayBegin, const CXXConstructExpr *e, bool newPointerIsChecked,
    bool zeroInitialize) {
  QualType elementType;
  mlir::Value numElements = emitArrayLength(arrayType, elementType, arrayBegin);
  emitCXXAggrConstructorCall(ctor, numElements, arrayBegin, e,
                             newPointerIsChecked, zeroInitialize);
}

/// Emit a loop to call a particular constructor for each of several members
/// of an array.
///
/// \param ctor the constructor to call for each element
/// \param numElements the number of elements in the array;
///   may be zero
/// \param arrayBase a T*, where T is the type constructed by ctor
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CIRGenFunction::emitCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, mlir::Value numElements, Address arrayBase,
    const CXXConstructExpr *e, bool newPointerIsChecked, bool zeroInitialize) {
  // It's legal for numElements to be zero.  This can happen both
  // dynamically, because x can be zero in 'new A[x]', and statically,
  // because of GCC extensions that permit zero-length arrays.  There
  // are probably legitimate places where we could assume that this
  // doesn't happen, but it's not clear that it's worth it.

  auto arrayTy = mlir::cast<cir::ArrayType>(arrayBase.getElementType());
  mlir::Type elementType = arrayTy.getElementType();

  // This might be a multi-dimensional array. Find the innermost element type.
  while (auto maybeArrayTy = mlir::dyn_cast<cir::ArrayType>(elementType))
    elementType = maybeArrayTy.getElementType();
  cir::PointerType ptrToElmType = builder.getPointerTo(elementType);

  // Optimize for a constant count.
  if (auto constantCount = numElements.getDefiningOp<cir::ConstantOp>()) {
    if (auto constIntAttr = constantCount.getValueAttr<cir::IntAttr>()) {
      // Just skip out if the constant count is zero.
      if (constIntAttr.getUInt() == 0)
        return;

      arrayTy = cir::ArrayType::get(elementType, constIntAttr.getUInt());
      // Otherwise, emit the check.
    }

    if (constantCount.use_empty())
      constantCount.erase();
  } else {
    // Otherwise, emit the check.
    cgm.errorNYI(e->getSourceRange(), "dynamic-length array expression");
  }

  // Tradional LLVM codegen emits a loop here. CIR lowers to a loop as part of
  // LoweringPrepare.

  // The alignment of the base, adjusted by the size of a single element,
  // provides a conservative estimate of the alignment of every element.
  // (This assumes we never start tracking offsetted alignments.)
  //
  // Note that these are complete objects and so we don't need to
  // use the non-virtual size or alignment.
  CanQualType type = getContext().getCanonicalTagType(ctor->getParent());
  CharUnits eltAlignment = arrayBase.getAlignment().alignmentOfArrayElement(
      getContext().getTypeSizeInChars(type));

  // Zero initialize the storage, if requested.
  if (zeroInitialize)
    emitNullInitialization(*currSrcLoc, arrayBase, type);

  // C++ [class.temporary]p4:
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array.
  // If the constructor has one or more default arguments, the destruction of
  // every temporary created in a default argument expression is sequenced
  // before the construction of the next array element, if any.
  {
    RunCleanupsScope scope(*this);

    // Evaluate the constructor and its arguments in a regular
    // partial-destroy cleanup.
    if (getLangOpts().Exceptions &&
        !ctor->getParent()->hasTrivialDestructor()) {
      cgm.errorNYI(e->getSourceRange(), "partial array cleanups");
    }

    // Emit the constructor call that will execute for every array element.
    mlir::Value arrayOp =
        builder.createPtrBitcast(arrayBase.getPointer(), arrayTy);
    builder.create<cir::ArrayCtor>(
        *currSrcLoc, arrayOp, [&](mlir::OpBuilder &b, mlir::Location loc) {
          mlir::BlockArgument arg =
              b.getInsertionBlock()->addArgument(ptrToElmType, loc);
          Address curAddr = Address(arg, elementType, eltAlignment);
          assert(!cir::MissingFeatures::sanitizers());
          auto currAVS = AggValueSlot::forAddr(
              curAddr, type.getQualifiers(), AggValueSlot::IsDestructed,
              AggValueSlot::IsNotAliased, AggValueSlot::DoesNotOverlap,
              AggValueSlot::IsNotZeroed);
          emitCXXConstructorCall(ctor, Ctor_Complete,
                                 /*ForVirtualBase=*/false,
                                 /*Delegating=*/false, currAVS, e);
          builder.create<cir::YieldOp>(loc);
        });
  }
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

void CIRGenFunction::emitImplicitAssignmentOperatorBody(FunctionArgList &args) {
  const auto *assignOp = cast<CXXMethodDecl>(curGD.getDecl());
  assert(assignOp->isCopyAssignmentOperator() ||
         assignOp->isMoveAssignmentOperator());
  const Stmt *rootS = assignOp->getBody();
  assert(isa<CompoundStmt>(rootS) &&
         "Body of an implicit assignment operator should be compound stmt.");
  const auto *rootCS = cast<CompoundStmt>(rootS);

  assert(!cir::MissingFeatures::incrementProfileCounter());
  assert(!cir::MissingFeatures::runCleanupsScope());

  // Classic codegen uses a special class to attempt to replace member
  // initializers with memcpy. We could possibly defer that to the
  // lowering or optimization phases to keep the memory accesses more
  // explicit. For now, we don't insert memcpy at all, though in some
  // cases the AST contains a call to memcpy.
  assert(!cir::MissingFeatures::assignMemcpyizer());
  for (Stmt *s : rootCS->body())
    if (emitStmt(s, /*useCurrentScope=*/true).failed())
      cgm.errorNYI(s->getSourceRange(),
                   std::string("emitImplicitAssignmentOperatorBody: ") +
                       s->getStmtClassName());
}

void CIRGenFunction::emitForwardingCallToLambda(
    const CXXMethodDecl *callOperator, CallArgList &callArgs) {
  // Get the address of the call operator.
  const CIRGenFunctionInfo &calleeFnInfo =
      cgm.getTypes().arrangeCXXMethodDeclaration(callOperator);
  cir::FuncOp calleePtr = cgm.getAddrOfFunction(
      GlobalDecl(callOperator), cgm.getTypes().getFunctionType(calleeFnInfo));

  // Prepare the return slot.
  const FunctionProtoType *fpt =
      callOperator->getType()->castAs<FunctionProtoType>();
  QualType resultType = fpt->getReturnType();
  ReturnValueSlot returnSlot;

  // We don't need to separately arrange the call arguments because
  // the call can't be variadic anyway --- it's impossible to forward
  // variadic arguments.

  // Now emit our call.
  CIRGenCallee callee =
      CIRGenCallee::forDirect(calleePtr, GlobalDecl(callOperator));
  RValue rv = emitCall(calleeFnInfo, callee, returnSlot, callArgs);

  // If necessary, copy the returned value into the slot.
  if (!resultType->isVoidType() && returnSlot.isNull()) {
    if (getLangOpts().ObjCAutoRefCount && resultType->isObjCRetainableType())
      cgm.errorNYI(callOperator->getSourceRange(),
                   "emitForwardingCallToLambda: ObjCAutoRefCount");
    emitReturnOfRValue(*currSrcLoc, rv, resultType);
  } else {
    cgm.errorNYI(callOperator->getSourceRange(),
                 "emitForwardingCallToLambda: return slot is not null");
  }
}

void CIRGenFunction::emitLambdaDelegatingInvokeBody(const CXXMethodDecl *md) {
  const CXXRecordDecl *lambda = md->getParent();

  // Start building arguments for forwarding call
  CallArgList callArgs;

  QualType lambdaType = getContext().getCanonicalTagType(lambda);
  QualType thisType = getContext().getPointerType(lambdaType);
  Address thisPtr =
      createMemTemp(lambdaType, getLoc(md->getSourceRange()), "unused.capture");
  callArgs.add(RValue::get(thisPtr.getPointer()), thisType);

  // Add the rest of the parameters.
  for (auto *param : md->parameters())
    emitDelegateCallArg(callArgs, param, param->getBeginLoc());

  const CXXMethodDecl *callOp = lambda->getLambdaCallOperator();
  // For a generic lambda, find the corresponding call operator specialization
  // to which the call to the static-invoker shall be forwarded.
  if (lambda->isGenericLambda()) {
    assert(md->isFunctionTemplateSpecialization());
    const TemplateArgumentList *tal = md->getTemplateSpecializationArgs();
    FunctionTemplateDecl *callOpTemplate =
        callOp->getDescribedFunctionTemplate();
    void *InsertPos = nullptr;
    FunctionDecl *correspondingCallOpSpecialization =
        callOpTemplate->findSpecialization(tal->asArray(), InsertPos);
    assert(correspondingCallOpSpecialization);
    callOp = cast<CXXMethodDecl>(correspondingCallOpSpecialization);
  }
  emitForwardingCallToLambda(callOp, callArgs);
}

void CIRGenFunction::emitLambdaStaticInvokeBody(const CXXMethodDecl *md) {
  if (md->isVariadic()) {
    // Codgen for LLVM doesn't emit code for this as well, it says:
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator
    // forward.
    cgm.errorNYI(md->getSourceRange(), "emitLambdaStaticInvokeBody: variadic");
  }

  emitLambdaDelegatingInvokeBody(md);
}

void CIRGenFunction::destroyCXXObject(CIRGenFunction &cgf, Address addr,
                                      QualType type) {
  const auto *record = type->castAsCXXRecordDecl();
  const CXXDestructorDecl *dtor = record->getDestructor();
  // TODO(cir): Unlike traditional codegen, CIRGen should actually emit trivial
  // dtors which shall be removed on later CIR passes. However, only remove this
  // assertion after we have a test case to exercise this path.
  assert(!dtor->isTrivial());
  cgf.emitCXXDestructorCall(dtor, Dtor_Complete, /*forVirtualBase*/ false,
                            /*delegating=*/false, addr, type);
}

namespace {
mlir::Value loadThisForDtorDelete(CIRGenFunction &cgf,
                                  const CXXDestructorDecl *dd) {
  if (Expr *thisArg = dd->getOperatorDeleteThisArg())
    return cgf.emitScalarExpr(thisArg);
  return cgf.loadCXXThis();
}

/// Call the operator delete associated with the current destructor.
struct CallDtorDelete final : EHScopeStack::Cleanup {
  CallDtorDelete() {}

  void emit(CIRGenFunction &cgf) override {
    const CXXDestructorDecl *dtor = cast<CXXDestructorDecl>(cgf.curFuncDecl);
    const CXXRecordDecl *classDecl = dtor->getParent();
    cgf.emitDeleteCall(dtor->getOperatorDelete(),
                       loadThisForDtorDelete(cgf, dtor),
                       cgf.getContext().getCanonicalTagType(classDecl));
  }
};

class DestroyField final : public EHScopeStack::Cleanup {
  const FieldDecl *field;
  CIRGenFunction::Destroyer *destroyer;

public:
  DestroyField(const FieldDecl *field, CIRGenFunction::Destroyer *destroyer)
      : field(field), destroyer(destroyer) {}

  void emit(CIRGenFunction &cgf) override {
    // Find the address of the field.
    Address thisValue = cgf.loadCXXThisAddress();
    CanQualType recordTy =
        cgf.getContext().getCanonicalTagType(field->getParent());
    LValue thisLV = cgf.makeAddrLValue(thisValue, recordTy);
    LValue lv = cgf.emitLValueForField(thisLV, field);
    assert(lv.isSimple());

    assert(!cir::MissingFeatures::ehCleanupFlags());
    cgf.emitDestroy(lv.getAddress(), field->getType(), destroyer);
  }
};
} // namespace

/// Emit all code that comes at the end of class's destructor. This is to call
/// destructors on members and base classes in reverse order of their
/// construction.
///
/// For a deleting destructor, this also handles the case where a destroying
/// operator delete completely overrides the definition.
void CIRGenFunction::enterDtorCleanups(const CXXDestructorDecl *dd,
                                       CXXDtorType dtorType) {
  assert((!dd->isTrivial() || dd->hasAttr<DLLExportAttr>()) &&
         "Should not emit dtor epilogue for non-exported trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (dtorType == Dtor_Deleting) {
    assert(dd->getOperatorDelete() &&
           "operator delete missing - EnterDtorCleanups");
    if (cxxStructorImplicitParamValue) {
      cgm.errorNYI(dd->getSourceRange(), "deleting destructor with vtt");
    } else {
      if (dd->getOperatorDelete()->isDestroyingOperatorDelete()) {
        cgm.errorNYI(dd->getSourceRange(),
                     "deleting destructor with destroying operator delete");
      } else {
        ehStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
      }
    }
    return;
  }

  const CXXRecordDecl *classDecl = dd->getParent();

  // Unions have no bases and do not call field destructors.
  if (classDecl->isUnion())
    return;

  // The complete-destructor phase just destructs all the virtual bases.
  if (dtorType == Dtor_Complete) {
    assert(!cir::MissingFeatures::sanitizers());

    // We push them in the forward order so that they'll be popped in
    // the reverse order.
    for (const CXXBaseSpecifier &base : classDecl->vbases()) {
      auto *baseClassDecl = base.getType()->castAsCXXRecordDecl();

      if (baseClassDecl->hasTrivialDestructor()) {
        // Under SanitizeMemoryUseAfterDtor, poison the trivial base class
        // memory. For non-trival base classes the same is done in the class
        // destructor.
        assert(!cir::MissingFeatures::sanitizers());
      } else {
        ehStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, baseClassDecl,
                                          /*baseIsVirtual=*/true);
      }
    }

    return;
  }

  assert(dtorType == Dtor_Base);
  assert(!cir::MissingFeatures::sanitizers());

  // Destroy non-virtual bases.
  for (const CXXBaseSpecifier &base : classDecl->bases()) {
    // Ignore virtual bases.
    if (base.isVirtual())
      continue;

    CXXRecordDecl *baseClassDecl = base.getType()->getAsCXXRecordDecl();

    if (baseClassDecl->hasTrivialDestructor())
      assert(!cir::MissingFeatures::sanitizers());
    else
      ehStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, baseClassDecl,
                                        /*baseIsVirtual=*/false);
  }

  assert(!cir::MissingFeatures::sanitizers());

  // Destroy direct fields.
  for (const FieldDecl *field : classDecl->fields()) {
    QualType type = field->getType();
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind)
      continue;

    // Anonymous union members do not have their destructors called.
    const RecordType *rt = type->getAsUnionType();
    if (rt && rt->getDecl()->isAnonymousStructOrUnion())
      continue;

    CleanupKind cleanupKind = getCleanupKind(dtorKind);
    assert(!cir::MissingFeatures::ehCleanupFlags());
    ehStack.pushCleanup<DestroyField>(cleanupKind, field,
                                      getDestroyer(dtorKind));
  }
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

void CIRGenFunction::emitCXXDestructorCall(const CXXDestructorDecl *dd,
                                           CXXDtorType type,
                                           bool forVirtualBase, bool delegating,
                                           Address thisAddr, QualType thisTy) {
  cgm.getCXXABI().emitDestructorCall(*this, dd, type, forVirtualBase,
                                     delegating, thisAddr, thisTy);
}

mlir::Value CIRGenFunction::getVTTParameter(GlobalDecl gd, bool forVirtualBase,
                                            bool delegating) {
  if (!cgm.getCXXABI().needsVTTParameter(gd))
    return nullptr;

  const CXXRecordDecl *rd = cast<CXXMethodDecl>(curCodeDecl)->getParent();
  const CXXRecordDecl *base = cast<CXXMethodDecl>(gd.getDecl())->getParent();

  uint64_t subVTTIndex;

  if (delegating) {
    // If this is a delegating constructor call, just load the VTT.
    return loadCXXVTT();
  } else if (rd == base) {
    // If the record matches the base, this is the complete ctor/dtor
    // variant calling the base variant in a class with virtual bases.
    assert(!cgm.getCXXABI().needsVTTParameter(curGD) &&
           "doing no-op VTT offset in base dtor/ctor?");
    assert(!forVirtualBase && "Can't have same class as virtual base!");
    subVTTIndex = 0;
  } else {
    const ASTRecordLayout &layout = getContext().getASTRecordLayout(rd);
    CharUnits baseOffset = forVirtualBase ? layout.getVBaseClassOffset(base)
                                          : layout.getBaseClassOffset(base);

    subVTTIndex =
        cgm.getVTables().getSubVTTIndex(rd, BaseSubobject(base, baseOffset));
    assert(subVTTIndex != 0 && "Sub-VTT index must be greater than zero!");
  }

  mlir::Location loc = cgm.getLoc(rd->getBeginLoc());
  if (cgm.getCXXABI().needsVTTParameter(curGD)) {
    // A VTT parameter was passed to the constructor, use it.
    mlir::Value vtt = loadCXXVTT();
    return builder.createVTTAddrPoint(loc, vtt.getType(), vtt, subVTTIndex);
  } else {
    // We're the complete constructor, so get the VTT by name.
    cir::GlobalOp vtt = cgm.getVTables().getAddrOfVTT(rd);
    return builder.createVTTAddrPoint(
        loc, builder.getPointerTo(cgm.VoidPtrTy),
        mlir::FlatSymbolRefAttr::get(vtt.getSymNameAttr()), subVTTIndex);
  }
}

Address CIRGenFunction::getAddressOfBaseClass(
    Address value, const CXXRecordDecl *derived,
    llvm::iterator_range<CastExpr::path_const_iterator> path,
    bool nullCheckValue, SourceLocation loc) {
  assert(!path.empty() && "Base path should not be empty!");

  CastExpr::path_const_iterator start = path.begin();
  const CXXRecordDecl *vBase = nullptr;

  if ((*path.begin())->isVirtual()) {
    vBase = (*start)->getType()->castAsCXXRecordDecl();
    ++start;
  }

  // Compute the static offset of the ultimate destination within its
  // allocating subobject (the virtual base, if there is one, or else
  // the "complete" object that we see).
  CharUnits nonVirtualOffset = cgm.computeNonVirtualBaseClassOffset(
      vBase ? vBase : derived, {start, path.end()});

  // If there's a virtual step, we can sometimes "devirtualize" it.
  // For now, that's limited to when the derived type is final.
  // TODO: "devirtualize" this for accesses to known-complete objects.
  if (vBase && derived->hasAttr<FinalAttr>()) {
    const ASTRecordLayout &layout = getContext().getASTRecordLayout(derived);
    CharUnits vBaseOffset = layout.getVBaseClassOffset(vBase);
    nonVirtualOffset += vBaseOffset;
    vBase = nullptr; // we no longer have a virtual step
  }

  // Get the base pointer type.
  mlir::Type baseValueTy = convertType((path.end()[-1])->getType());
  assert(!cir::MissingFeatures::addressSpace());

  // If there is no virtual base, use cir.base_class_addr.  It takes care of
  // the adjustment and the null pointer check.
  if (nonVirtualOffset.isZero() && !vBase) {
    assert(!cir::MissingFeatures::sanitizers());
    return builder.createBaseClassAddr(getLoc(loc), value, baseValueTy, 0,
                                       /*assumeNotNull=*/true);
  }

  assert(!cir::MissingFeatures::sanitizers());

  // Compute the virtual offset.
  mlir::Value virtualOffset = nullptr;
  if (vBase) {
    virtualOffset = cgm.getCXXABI().getVirtualBaseClassOffset(
        getLoc(loc), *this, value, derived, vBase);
  }

  // Apply both offsets.
  value = applyNonVirtualAndVirtualOffset(
      getLoc(loc), *this, value, nonVirtualOffset, virtualOffset, derived,
      vBase, baseValueTy, not nullCheckValue);

  // Cast to the destination type.
  value = value.withElementType(builder, baseValueTy);

  return value;
}

// TODO(cir): this can be shared with LLVM codegen.
bool CIRGenFunction::shouldEmitVTableTypeCheckedLoad(const CXXRecordDecl *rd) {
  assert(!cir::MissingFeatures::hiddenVisibility());
  if (!cgm.getCodeGenOpts().WholeProgramVTables)
    return false;

  if (cgm.getCodeGenOpts().VirtualFunctionElimination)
    return true;

  assert(!cir::MissingFeatures::sanitizers());

  return false;
}

mlir::Value CIRGenFunction::getVTablePtr(mlir::Location loc, Address thisAddr,
                                         const CXXRecordDecl *rd) {
  auto vtablePtr = cir::VTableGetVPtrOp::create(
      builder, loc, builder.getPtrToVPtrType(), thisAddr.getPointer());
  Address vtablePtrAddr = Address(vtablePtr, thisAddr.getAlignment());

  auto vtable = builder.createLoad(loc, vtablePtrAddr);
  assert(!cir::MissingFeatures::opTBAA());

  if (cgm.getCodeGenOpts().OptimizationLevel > 0 &&
      cgm.getCodeGenOpts().StrictVTablePointers) {
    assert(!cir::MissingFeatures::createInvariantGroup());
  }

  return vtable;
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
  CIRGenCXXABI::AddedStructorArgCounts extraArgs =
      cgm.getCXXABI().addImplicitConstructorArgs(*this, d, type, forVirtualBase,
                                                 delegating, args);

  // Emit the call.
  auto calleePtr = cgm.getAddrOfCXXStructor(GlobalDecl(d, type));
  const CIRGenFunctionInfo &info = cgm.getTypes().arrangeCXXConstructorCall(
      args, d, type, extraArgs.prefix, extraArgs.suffix, passPrototypeArgs);
  CIRGenCallee callee = CIRGenCallee::forDirect(calleePtr, GlobalDecl(d, type));
  cir::CIRCallOpInterface c;
  emitCall(info, callee, ReturnValueSlot(), args, &c, getLoc(loc));

  if (cgm.getCodeGenOpts().OptimizationLevel != 0 && !crd->isDynamicClass() &&
      type != Ctor_Base && cgm.getCodeGenOpts().StrictVTablePointers)
    cgm.errorNYI(d->getSourceRange(), "vtable assumption loads");
}
