//===--- CIRGenClass.cpp - Emit CIR Code for C++ classes --------*- C++ -*-===//
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
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace cir;

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delgation optimization, i.e. emitting the
/// complete constructor as a simple call to the base constructor.
bool CIRGenFunction::IsConstructorDelegationValid(
    const CXXConstructorDecl *Ctor) {

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
  if (Ctor->getParent()->getNumVBases()) {
    llvm_unreachable("NYI");
  }

  // We also disable the optimization for variadic functions because it's
  // impossible to "re-pass" varargs.
  if (Ctor->getType()->castAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (Ctor->isDelegatingConstructor())
    llvm_unreachable("NYI");

  return true;
}

namespace {
class FieldMemcpyizer {
public:
  FieldMemcpyizer(CIRGenFunction &CGF, const CXXRecordDecl *ClassDecl,
                  const VarDecl *SrcRec)
      : CGF(CGF), ClassDecl(ClassDecl),
        // SrcRec(SrcRec),
        RecLayout(CGF.getContext().getASTRecordLayout(ClassDecl)),
        FirstField(nullptr), LastField(nullptr), FirstFieldOffset(0),
        LastFieldOffset(0), LastAddedFieldIndex(0) {
    (void)SrcRec;
  }

  bool isMemcpyableField(FieldDecl *F) const {
    // Never memcpy fields when we are adding poised paddings.
    if (CGF.getContext().getLangOpts().SanitizeAddressFieldPadding)
      return false;
    Qualifiers Qual = F->getType().getQualifiers();
    if (Qual.hasVolatile() || Qual.hasObjCLifetime())
      return false;

    return true;
  }

  void addMemcpyableField(FieldDecl *F) {
    if (F->isZeroSize(CGF.getContext()))
      return;
    if (!FirstField)
      addInitialField(F);
    else
      addNextField(F);
  }

  CharUnits getMemcpySize(uint64_t FirstByteOffset) const {
    ASTContext &Ctx = CGF.getContext();
    unsigned LastFieldSize =
        LastField->isBitField()
            ? LastField->getBitWidthValue(Ctx)
            : Ctx.toBits(
                  Ctx.getTypeInfoDataSizeInChars(LastField->getType()).Width);
    uint64_t MemcpySizeBits = LastFieldOffset + LastFieldSize -
                              FirstByteOffset + Ctx.getCharWidth() - 1;
    CharUnits MemcpySize = Ctx.toCharUnitsFromBits(MemcpySizeBits);
    return MemcpySize;
  }

  void buildMemcpy() {
    // Give the subclass a chance to bail out if it feels the memcpy isn't worth
    // it (e.g. Hasn't aggregated enough data).
    if (!FirstField) {
      return;
    }

    llvm_unreachable("NYI");
  }

  void reset() { FirstField = nullptr; }

protected:
  CIRGenFunction &CGF;
  const CXXRecordDecl *ClassDecl;

private:
  void buildMemcpyIR(Address DestPtr, Address SrcPtr, CharUnits Size) {
    llvm_unreachable("NYI");
  }

  void addInitialField(FieldDecl *F) {
    FirstField = F;
    LastField = F;
    FirstFieldOffset = RecLayout.getFieldOffset(F->getFieldIndex());
    LastFieldOffset = FirstFieldOffset;
    LastAddedFieldIndex = F->getFieldIndex();
  }

  void addNextField(FieldDecl *F) {
    // For the most part, the following invariant will hold:
    //   F->getFieldIndex() == LastAddedFieldIndex + 1
    // The one exception is that Sema won't add a copy-initializer for an
    // unnamed bitfield, which will show up here as a gap in the sequence.
    assert(F->getFieldIndex() >= LastAddedFieldIndex + 1 &&
           "Cannot aggregate fields out of order.");
    LastAddedFieldIndex = F->getFieldIndex();

    // The 'first' and 'last' fields are chosen by offset, rather than field
    // index. This allows the code to support bitfields, as well as regular
    // fields.
    uint64_t FOffset = RecLayout.getFieldOffset(F->getFieldIndex());
    if (FOffset < FirstFieldOffset) {
      FirstField = F;
      FirstFieldOffset = FOffset;
    } else if (FOffset >= LastFieldOffset) {
      LastField = F;
      LastFieldOffset = FOffset;
    }
  }

  // const VarDecl *SrcRec;
  const ASTRecordLayout &RecLayout;
  FieldDecl *FirstField;
  FieldDecl *LastField;
  uint64_t FirstFieldOffset, LastFieldOffset;
  unsigned LastAddedFieldIndex;
};

static void buildLValueForAnyFieldInitialization(CIRGenFunction &CGF,
                                                 CXXCtorInitializer *MemberInit,
                                                 LValue &LHS) {
  FieldDecl *Field = MemberInit->getAnyMember();
  if (MemberInit->isIndirectMemberInitializer()) {
    llvm_unreachable("NYI");
  } else {
    LHS = CGF.buildLValueForFieldInitialization(LHS, Field);
  }
}

static void buildMemberInitializer(CIRGenFunction &CGF,
                                   const CXXRecordDecl *ClassDecl,
                                   CXXCtorInitializer *MemberInit,
                                   const CXXConstructorDecl *Constructor,
                                   FunctionArgList &Args) {
  // TODO: ApplyDebugLocation
  assert(MemberInit->isAnyMemberInitializer() &&
         "Mush have member initializer!");
  assert(MemberInit->getInit() && "Must have initializer!");

  // non-static data member initializers
  FieldDecl *Field = MemberInit->getAnyMember();
  QualType FieldType = Field->getType();

  mlir::Operation *ThisPtr = CGF.LoadCXXThis();
  QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
  LValue LHS;

  // If a base constructor is being emitted, create an LValue that has the
  // non-virtual alignment.
  if (CGF.CurGD.getCtorType() == Ctor_Base)
    LHS = CGF.MakeNaturalAlignPointeeAddrLValue(ThisPtr, RecordTy);
  else
    llvm_unreachable("NYI");

  buildLValueForAnyFieldInitialization(CGF, MemberInit, LHS);

  // Special case: If we are in a copy or move constructor, and we are copying
  // an array off PODs or classes with tirival copy constructors, ignore the AST
  // and perform the copy we know is equivalent.
  // FIXME: This is hacky at best... if we had a bit more explicit information
  // in the AST, we could generalize it more easily.
  const ConstantArrayType *Array =
      CGF.getContext().getAsConstantArrayType(FieldType);
  if (Array && Constructor->isDefaulted() &&
      Constructor->isCopyOrMoveConstructor()) {
    llvm_unreachable("NYI");
  }

  CGF.buildInitializerForField(Field, LHS, MemberInit->getInit());
}

class ConstructorMemcpyizer : public FieldMemcpyizer {
private:
  /// Get source argument for copy constructor. Returns null if not a copy
  /// constructor.
  static const VarDecl *getTrivialCopySource(CIRGenFunction &CGF,
                                             const CXXConstructorDecl *CD,
                                             FunctionArgList &Args) {
    if (CD->isCopyOrMoveConstructor() && CD->isDefaulted())
      llvm_unreachable("NYI");

    return nullptr;
  }

  // Returns true if a CXXCtorInitializer represents a member initialization
  // that can be rolled into a memcpy
  bool isMemberInitMemcpyable(CXXCtorInitializer *MemberInit) const {
    if (!MemcpyableCtor)
      return false;

    llvm_unreachable("NYI");
  }

public:
  ConstructorMemcpyizer(CIRGenFunction &CGF, const CXXConstructorDecl *CD,
                        FunctionArgList &Args)
      : FieldMemcpyizer(CGF, CD->getParent(),
                        getTrivialCopySource(CGF, CD, Args)),
        ConstructorDecl(CD),
        MemcpyableCtor(CD->isDefaulted() && CD->isCopyOrMoveConstructor() &&
                       CGF.getLangOpts().getGC() == LangOptions::NonGC),
        Args(Args) {}

  void addMemberInitializer(CXXCtorInitializer *MemberInit) {
    if (isMemberInitMemcpyable(MemberInit)) {
      llvm_unreachable("NYI");
    } else {
      buildAggregatedInits();
      buildMemberInitializer(CGF, ConstructorDecl->getParent(), MemberInit,
                             ConstructorDecl, Args);
    }
  }

  void buildAggregatedInits() {
    if (AggregatedInits.size() <= 1) {
      // This memcpy is too small to be worthwhile. Fall back on default
      // codegen.
      if (!AggregatedInits.empty()) {
        llvm_unreachable("NYI");
      }
      reset();
      return;
    }

    pushEHDestructors();
    buildMemcpy();
    AggregatedInits.clear();
  }

  void pushEHDestructors() {
    Address ThisPtr = CGF.LoadCXXThisAddress();
    QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
    LValue LHS = CGF.makeAddrLValue(ThisPtr, RecordTy);
    (void)LHS;

    for (unsigned i = 0; i < AggregatedInits.size(); ++i) {
      llvm_unreachable("NYI");
    }
  }

  void finish() { buildAggregatedInits(); }

private:
  const CXXConstructorDecl *ConstructorDecl;
  bool MemcpyableCtor;
  FunctionArgList &Args;
  SmallVector<CXXCtorInitializer *, 16> AggregatedInits;
};

} // namespace

/// buildCtorPrologue - This routine generates necessary code to initialize base
/// classes and non-static data members belonging to this constructor.
void CIRGenFunction::buildCtorPrologue(const CXXConstructorDecl *CD,
                                       CXXCtorType CtorType,
                                       FunctionArgList &Args) {
  if (CD->isDelegatingConstructor())
    llvm_unreachable("NYI");

  const CXXRecordDecl *ClassDecl = CD->getParent();

  CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
                                          E = CD->init_end();

  // Virtual base initializers first, if any. They aren't needed if:
  // - This is a base ctor variant
  // - There are no vbases
  // - The class is abstract, so a complete object of it cannot be constructed
  //
  // The check for an abstract class is necessary because sema may not have
  // marked virtual base destructors referenced.
  bool ConstructVBases = CtorType != Ctor_Base &&
                         ClassDecl->getNumVBases() != 0 &&
                         !ClassDecl->isAbstract();

  // In the Microsoft C++ ABI, there are no constructor variants. Instead, the
  // constructor of a class with virtual bases takes an additional parameter to
  // conditionally construct the virtual bases. Emit that check here.
  mlir::Block *BaseCtorContinueBB = nullptr;
  if (ConstructVBases &&
      !CGM.getTarget().getCXXABI().hasConstructorVariants()) {
    llvm_unreachable("NYI");
  }

  mlir::Operation *const OldThis = CXXThisValue;
  for (; B != E && (*B)->isBaseInitializer() && (*B)->isBaseVirtual(); B++) {
    if (!ConstructVBases)
      continue;
    llvm_unreachable("NYI");
  }

  if (BaseCtorContinueBB) {
    llvm_unreachable("NYI");
  }

  // Then, non-virtual base initializers.
  for (; B != E && (*B)->isBaseInitializer(); B++) {
    assert(!(*B)->isBaseVirtual());

    if (CGM.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");

    llvm_unreachable("NYI");
  }

  CXXThisValue = OldThis;

  initializeVTablePointers(ClassDecl);

  // And finally, initialize class members.
  FieldConstructionScope FCS(*this, LoadCXXThisAddress());
  ConstructorMemcpyizer CM(*this, CD, Args);
  for (; B != E; B++) {
    CXXCtorInitializer *Member = (*B);
    assert(!Member->isBaseInitializer());
    assert(Member->isAnyMemberInitializer() &&
           "Delegating initializer on non-delegating constructor");
    CM.addMemberInitializer(Member);
  }
  CM.finish();
}

void CIRGenFunction::initializeVTablePointers(const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  if (CGM.getCXXABI().doStructorsInitializeVPtrs(RD))
    for (const auto &Vptr : getVTablePointers(RD)) {
      llvm_unreachable("NYI");
      (void)Vptr;
    }

  if (RD->getNumVBases())
    llvm_unreachable("NYI");
}

CIRGenFunction::VPtrsVector
CIRGenFunction::getVTablePointers(const CXXRecordDecl *VTableClass) {
  CIRGenFunction::VPtrsVector VPtrsResult;
  VisitedVirtualBasesSetTy VBases;
  getVTablePointers(BaseSubobject(VTableClass, CharUnits::Zero()),
                    /*NearestVBase=*/nullptr,
                    /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                    /*BaseIsNonVirtualPrimaryBase=*/false, VTableClass, VBases,
                    VPtrsResult);
  return VPtrsResult;
}

void CIRGenFunction::getVTablePointers(BaseSubobject Base,
                                       const CXXRecordDecl *NearestVBase,
                                       CharUnits OffsetFromNearestVBase,
                                       bool BaseIsNonVirtualPrimaryBase,
                                       const CXXRecordDecl *VTableClass,
                                       VisitedVirtualBasesSetTy &VBases,
                                       VPtrsVector &Vptrs) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    VPtr Vptr = {Base, NearestVBase, OffsetFromNearestVBase, VTableClass};
    Vptrs.push_back(Vptr);
  }

  const CXXRecordDecl *RD = Base.getBase();

  // Traverse bases.
  for (const auto &I : RD->bases()) {
    (void)I;
    llvm_unreachable("NYI");
  }
}

Address CIRGenFunction::LoadCXXThisAddress() {
  assert(CurFuncDecl && "loading 'this' without a func declaration?");
  assert(isa<CXXMethodDecl>(CurFuncDecl));

  // Lazily compute CXXThisAlignment.
  if (CXXThisAlignment.isZero()) {
    // Just use the best known alignment for the parent.
    // TODO: if we're currently emitting a complete-object ctor/dtor, we can
    // always use the complete-object alignment.
    auto RD = cast<CXXMethodDecl>(CurFuncDecl)->getParent();
    CXXThisAlignment = CGM.getClassPointerAlignment(RD);
  }

  // Consider how to do this if we ever have multiple returns
  auto Result = LoadCXXThis()->getOpResult(0);
  return Address(Result, CXXThisAlignment);
}

void CIRGenFunction::buildInitializerForField(FieldDecl *Field, LValue LHS,
                                              Expr *Init) {
  llvm_unreachable("NYI");
  QualType FieldType = Field->getType();

  // Ensure that we destroy this object if an exception is thrown later in the
  // constructor.
  QualType::DestructionKind dtorKind = FieldType.isDestructedType();
  (void)dtorKind;
  if (UnimplementedFeature::cleanups())
    llvm_unreachable("NYI");
}

void CIRGenFunction::buildDelegateCXXConstructorCall(
    const CXXConstructorDecl *Ctor, CXXCtorType CtorType,
    const FunctionArgList &Args, SourceLocation Loc) {
  CallArgList DelegateArgs;

  FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
  assert(I != E && "no parameters to constructor");

  // this
  Address This = LoadCXXThisAddress();
  DelegateArgs.add(RValue::get(This.getPointer()), (*I)->getType());
  ++I;

  // FIXME: The location of the VTT parameter in the parameter list is specific
  // to the Itanium ABI and shouldn't be hardcoded here.
  if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
    llvm_unreachable("NYI");
  }

  // Explicit arguments.
  for (; I != E; ++I) {
    const VarDecl *param = *I;
    // FIXME: per-argument source location
    buildDelegateCallArg(DelegateArgs, param, Loc);
  }

  buildCXXConstructorCall(Ctor, CtorType, /*ForVirtualBase=*/false,
                          /*Delegating=*/true, This, DelegateArgs,
                          AggValueSlot::MayOverlap, Loc,
                          /*NewPointerIsChecked=*/true);
}
