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

#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/TargetBuiltins.h"

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
  if (Ctor->getParent()->getNumVBases())
    return false;

  // We also disable the optimization for variadic functions because it's
  // impossible to "re-pass" varargs.
  if (Ctor->getType()->castAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (Ctor->isDelegatingConstructor())
    llvm_unreachable("NYI");

  return true;
}

/// TODO(cir): strong candidate for AST helper to be shared between LLVM and CIR
/// codegen.
static bool isMemcpyEquivalentSpecialMember(const CXXMethodDecl *D) {
  auto *CD = dyn_cast<CXXConstructorDecl>(D);
  if (!(CD && CD->isCopyOrMoveConstructor()) &&
      !D->isCopyAssignmentOperator() && !D->isMoveAssignmentOperator())
    return false;

  // We can emit a memcpy for a trivial copy or move constructor/assignment.
  if (D->isTrivial() && !D->getParent()->mayInsertExtraPadding())
    return true;

  // We *must* emit a memcpy for a defaulted union copy or move op.
  if (D->getParent()->isUnion() && D->isDefaulted())
    return true;

  return false;
}

namespace {
/// TODO(cir): a lot of what we see under this namespace is a strong candidate
/// to be shared between LLVM and CIR codegen.

/// RAII object to indicate that codegen is copying the value representation
/// instead of the object representation. Useful when copying a struct or
/// class which has uninitialized members and we're only performing
/// lvalue-to-rvalue conversion on the object but not its members.
class CopyingValueRepresentation {
public:
  explicit CopyingValueRepresentation(CIRGenFunction &CGF)
      : CGF(CGF), OldSanOpts(CGF.SanOpts) {
    CGF.SanOpts.set(SanitizerKind::Bool, false);
    CGF.SanOpts.set(SanitizerKind::Enum, false);
  }
  ~CopyingValueRepresentation() { CGF.SanOpts = OldSanOpts; }

private:
  CIRGenFunction &CGF;
  SanitizerSet OldSanOpts;
};

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
    LHS = CGF.buildLValueForFieldInitialization(LHS, Field, Field->getName());
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

  auto ThisPtr = CGF.LoadCXXThis();
  QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
  LValue LHS;

  // If a base constructor is being emitted, create an LValue that has the
  // non-virtual alignment.
  if (CGF.CurGD.getCtorType() == Ctor_Base)
    LHS = CGF.MakeNaturalAlignPointeeAddrLValue(ThisPtr, RecordTy);
  else
    LHS = CGF.MakeNaturalAlignAddrLValue(ThisPtr, RecordTy);

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
      return Args[CGF.CGM.getCXXABI().getSrcArgforCopyCtor(CD, Args)];

    return nullptr;
  }

  // Returns true if a CXXCtorInitializer represents a member initialization
  // that can be rolled into a memcpy.
  bool isMemberInitMemcpyable(CXXCtorInitializer *MemberInit) const {
    if (!MemcpyableCtor)
      return false;

    assert(!UnimplementedFeature::fieldMemcpyizerBuildMemcpy());
    return false;
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
      AggregatedInits.push_back(MemberInit);
      addMemcpyableField(MemberInit->getMember());
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
      CXXCtorInitializer *MemberInit = AggregatedInits[i];
      QualType FieldType = MemberInit->getAnyMember()->getType();
      QualType::DestructionKind dtorKind = FieldType.isDestructedType();
      if (!CGF.needsEHCleanup(dtorKind))
        continue;
      LValue FieldLHS = LHS;
      buildLValueForAnyFieldInitialization(CGF, MemberInit, FieldLHS);
      CGF.pushEHDestroy(dtorKind, FieldLHS.getAddress(), FieldType);
    }
  }

  void finish() { buildAggregatedInits(); }

private:
  const CXXConstructorDecl *ConstructorDecl;
  bool MemcpyableCtor;
  FunctionArgList &Args;
  SmallVector<CXXCtorInitializer *, 16> AggregatedInits;
};

class AssignmentMemcpyizer : public FieldMemcpyizer {
private:
  // Returns the memcpyable field copied by the given statement, if one
  // exists. Otherwise returns null.
  FieldDecl *getMemcpyableField(Stmt *S) {
    if (!AssignmentsMemcpyable)
      return nullptr;
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(S)) {
      // Recognise trivial assignments.
      if (BO->getOpcode() != BO_Assign)
        return nullptr;
      MemberExpr *ME = dyn_cast<MemberExpr>(BO->getLHS());
      if (!ME)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      Stmt *RHS = BO->getRHS();
      if (ImplicitCastExpr *EC = dyn_cast<ImplicitCastExpr>(RHS))
        RHS = EC->getSubExpr();
      if (!RHS)
        return nullptr;
      if (MemberExpr *ME2 = dyn_cast<MemberExpr>(RHS)) {
        if (ME2->getMemberDecl() == Field)
          return Field;
      }
      return nullptr;
    } else if (CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(S)) {
      CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MCE->getCalleeDecl());
      if (!(MD && isMemcpyEquivalentSpecialMember(MD)))
        return nullptr;
      MemberExpr *IOA = dyn_cast<MemberExpr>(MCE->getImplicitObjectArgument());
      if (!IOA)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(IOA->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      MemberExpr *Arg0 = dyn_cast<MemberExpr>(MCE->getArg(0));
      if (!Arg0 || Field != dyn_cast<FieldDecl>(Arg0->getMemberDecl()))
        return nullptr;
      return Field;
    } else if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
      FunctionDecl *FD = dyn_cast<FunctionDecl>(CE->getCalleeDecl());
      if (!FD || FD->getBuiltinID() != Builtin::BI__builtin_memcpy)
        return nullptr;
      Expr *DstPtr = CE->getArg(0);
      if (ImplicitCastExpr *DC = dyn_cast<ImplicitCastExpr>(DstPtr))
        DstPtr = DC->getSubExpr();
      UnaryOperator *DUO = dyn_cast<UnaryOperator>(DstPtr);
      if (!DUO || DUO->getOpcode() != UO_AddrOf)
        return nullptr;
      MemberExpr *ME = dyn_cast<MemberExpr>(DUO->getSubExpr());
      if (!ME)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      Expr *SrcPtr = CE->getArg(1);
      if (ImplicitCastExpr *SC = dyn_cast<ImplicitCastExpr>(SrcPtr))
        SrcPtr = SC->getSubExpr();
      UnaryOperator *SUO = dyn_cast<UnaryOperator>(SrcPtr);
      if (!SUO || SUO->getOpcode() != UO_AddrOf)
        return nullptr;
      MemberExpr *ME2 = dyn_cast<MemberExpr>(SUO->getSubExpr());
      if (!ME2 || Field != dyn_cast<FieldDecl>(ME2->getMemberDecl()))
        return nullptr;
      return Field;
    }

    return nullptr;
  }

  bool AssignmentsMemcpyable;
  SmallVector<Stmt *, 16> AggregatedStmts;

public:
  AssignmentMemcpyizer(CIRGenFunction &CGF, const CXXMethodDecl *AD,
                       FunctionArgList &Args)
      : FieldMemcpyizer(CGF, AD->getParent(), Args[Args.size() - 1]),
        AssignmentsMemcpyable(CGF.getLangOpts().getGC() == LangOptions::NonGC) {
    assert(Args.size() == 2);
  }

  void emitAssignment(Stmt *S) {
    FieldDecl *F = getMemcpyableField(S);
    if (F) {
      addMemcpyableField(F);
      AggregatedStmts.push_back(S);
    } else {
      emitAggregatedStmts();
      if (CGF.buildStmt(S, /*useCurrentScope=*/true).failed())
        llvm_unreachable("Should not get here!");
    }
  }

  void emitAggregatedStmts() {
    if (AggregatedStmts.size() <= 1) {
      if (!AggregatedStmts.empty()) {
        CopyingValueRepresentation CVR(CGF);
        if (CGF.buildStmt(AggregatedStmts[0], /*useCurrentScope=*/true)
                .failed())
          llvm_unreachable("Should not get here!");
      }
      reset();
    }

    buildMemcpy();
    AggregatedStmts.clear();
  }

  void finish() { emitAggregatedStmts(); }
};
} // namespace

static bool isInitializerOfDynamicClass(const CXXCtorInitializer *BaseInit) {
  const Type *BaseType = BaseInit->getBaseClass();
  const auto *BaseClassDecl =
      cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());
  return BaseClassDecl->isDynamicClass();
}

namespace {
/// Call the destructor for a direct base class.
struct CallBaseDtor final : EHScopeStack::Cleanup {
  const CXXRecordDecl *BaseClass;
  bool BaseIsVirtual;
  CallBaseDtor(const CXXRecordDecl *Base, bool BaseIsVirtual)
      : BaseClass(Base), BaseIsVirtual(BaseIsVirtual) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    const CXXRecordDecl *DerivedClass =
        cast<CXXMethodDecl>(CGF.CurCodeDecl)->getParent();

    const CXXDestructorDecl *D = BaseClass->getDestructor();
    // We are already inside a destructor, so presumably the object being
    // destroyed should have the expected type.
    QualType ThisTy = D->getFunctionObjectParameterType();
    assert(CGF.currSrcLoc && "expected source location");
    Address Addr = CGF.getAddressOfDirectBaseInCompleteClass(
        *CGF.currSrcLoc, CGF.LoadCXXThisAddress(), DerivedClass, BaseClass,
        BaseIsVirtual);
    CGF.buildCXXDestructorCall(D, Dtor_Base, BaseIsVirtual,
                               /*Delegating=*/false, Addr, ThisTy);
  }
};

/// A visitor which checks whether an initializer uses 'this' in a
/// way which requires the vtable to be properly set.
struct DynamicThisUseChecker
    : ConstEvaluatedExprVisitor<DynamicThisUseChecker> {
  typedef ConstEvaluatedExprVisitor<DynamicThisUseChecker> super;

  bool UsesThis;

  DynamicThisUseChecker(const ASTContext &C) : super(C), UsesThis(false) {}

  // Black-list all explicit and implicit references to 'this'.
  //
  // Do we need to worry about external references to 'this' derived
  // from arbitrary code?  If so, then anything which runs arbitrary
  // external code might potentially access the vtable.
  void VisitCXXThisExpr(const CXXThisExpr *E) { UsesThis = true; }
};
} // end anonymous namespace

static bool BaseInitializerUsesThis(ASTContext &C, const Expr *Init) {
  DynamicThisUseChecker Checker(C);
  Checker.Visit(Init);
  return Checker.UsesThis;
}

/// Gets the address of a direct base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'This' is assumed to be non-null.
Address CIRGenFunction::getAddressOfDirectBaseInCompleteClass(
    mlir::Location loc, Address This, const CXXRecordDecl *Derived,
    const CXXRecordDecl *Base, bool BaseIsVirtual) {
  // 'this' must be a pointer (in some address space) to Derived.
  assert(This.getElementType() == ConvertType(Derived));

  // Compute the offset of the virtual base.
  CharUnits Offset;
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(Derived);
  if (BaseIsVirtual)
    Offset = Layout.getVBaseClassOffset(Base);
  else
    Offset = Layout.getBaseClassOffset(Base);

  // Shift and cast down to the base type.
  // TODO: for complete types, this should be possible with a GEP.
  Address V = This;
  if (!Offset.isZero()) {
    mlir::Value OffsetVal = builder.getSInt32(Offset.getQuantity(), loc);
    mlir::Value VBaseThisPtr = builder.create<mlir::cir::PtrStrideOp>(
        loc, This.getPointer().getType(), This.getPointer(), OffsetVal);
    V = Address(VBaseThisPtr, CXXABIThisAlignment);
  }
  V = builder.createElementBitCast(loc, V, ConvertType(Base));
  return V;
}

static void buildBaseInitializer(mlir::Location loc, CIRGenFunction &CGF,
                                 const CXXRecordDecl *ClassDecl,
                                 CXXCtorInitializer *BaseInit) {
  assert(BaseInit->isBaseInitializer() && "Must have base initializer!");

  Address ThisPtr = CGF.LoadCXXThisAddress();

  const Type *BaseType = BaseInit->getBaseClass();
  const auto *BaseClassDecl =
      cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());

  bool isBaseVirtual = BaseInit->isBaseVirtual();

  // If the initializer for the base (other than the constructor
  // itself) accesses 'this' in any way, we need to initialize the
  // vtables.
  if (BaseInitializerUsesThis(CGF.getContext(), BaseInit->getInit()))
    CGF.initializeVTablePointers(loc, ClassDecl);

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  Address V = CGF.getAddressOfDirectBaseInCompleteClass(
      loc, ThisPtr, ClassDecl, BaseClassDecl, isBaseVirtual);
  AggValueSlot AggSlot = AggValueSlot::forAddr(
      V, Qualifiers(), AggValueSlot::IsDestructed,
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
      CGF.getOverlapForBaseInit(ClassDecl, BaseClassDecl, isBaseVirtual));

  CGF.buildAggExpr(BaseInit->getInit(), AggSlot);

  if (CGF.CGM.getLangOpts().Exceptions &&
      !BaseClassDecl->hasTrivialDestructor())
    CGF.EHStack.pushCleanup<CallBaseDtor>(EHCleanup, BaseClassDecl,
                                          isBaseVirtual);
}

/// This routine generates necessary code to initialize base classes and
/// non-static data members belonging to this constructor.
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

  auto const OldThis = CXXThisValue;
  for (; B != E && (*B)->isBaseInitializer() && (*B)->isBaseVirtual(); B++) {
    if (!ConstructVBases)
      continue;
    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        CGM.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(*B))
      llvm_unreachable("NYI");
    buildBaseInitializer(getLoc(CD->getBeginLoc()), *this, ClassDecl, *B);
  }

  if (BaseCtorContinueBB) {
    llvm_unreachable("NYI");
  }

  // Then, non-virtual base initializers.
  for (; B != E && (*B)->isBaseInitializer(); B++) {
    assert(!(*B)->isBaseVirtual());

    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        CGM.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(*B))
      llvm_unreachable("NYI");
    buildBaseInitializer(getLoc(CD->getBeginLoc()), *this, ClassDecl, *B);
  }

  CXXThisValue = OldThis;

  initializeVTablePointers(getLoc(CD->getBeginLoc()), ClassDecl);

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

static Address ApplyNonVirtualAndVirtualOffset(
    CIRGenFunction &CGF, Address addr, CharUnits nonVirtualOffset,
    mlir::Value virtualOffset, const CXXRecordDecl *derivedClass,
    const CXXRecordDecl *nearestVBase) {
  llvm_unreachable("NYI");
  return Address::invalid();
}

void CIRGenFunction::initializeVTablePointer(mlir::Location loc,
                                             const VPtr &Vptr) {
  // Compute the address point.
  auto VTableAddressPoint = CGM.getCXXABI().getVTableAddressPointInStructor(
      *this, Vptr.VTableClass, Vptr.Base, Vptr.NearestVBase);

  if (!VTableAddressPoint)
    return;

  // Compute where to store the address point.
  mlir::Value VirtualOffset{};
  CharUnits NonVirtualOffset = CharUnits::Zero();

  if (CGM.getCXXABI().isVirtualOffsetNeededForVTableField(*this, Vptr)) {
    llvm_unreachable("NYI");
  } else {
    // We can just use the base offset in the complete class.
    NonVirtualOffset = Vptr.Base.getBaseOffset();
  }

  // Apply the offsets.
  Address VTableField = LoadCXXThisAddress();
  if (!NonVirtualOffset.isZero() || VirtualOffset) {
    VTableField = ApplyNonVirtualAndVirtualOffset(
        *this, VTableField, NonVirtualOffset, VirtualOffset, Vptr.VTableClass,
        Vptr.NearestVBase);
  }

  // Finally, store the address point. Use the same CIR types as the field.
  //
  // vtable field is derived from `this` pointer, therefore they should be in
  // the same addr space.
  assert(!UnimplementedFeature::addressSpace());
  VTableField = builder.createElementBitCast(loc, VTableField,
                                             VTableAddressPoint.getType());
  builder.createStore(loc, VTableAddressPoint, VTableField);
  assert(!UnimplementedFeature::tbaa());
}

void CIRGenFunction::initializeVTablePointers(mlir::Location loc,
                                              const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  if (CGM.getCXXABI().doStructorsInitializeVPtrs(RD))
    for (const auto &Vptr : getVTablePointers(RD))
      initializeVTablePointer(loc, Vptr);

  if (RD->getNumVBases())
    CGM.getCXXABI().initializeHiddenVirtualInheritanceMembers(*this, RD);
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
    auto *BaseDecl =
        cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());

    // Ignore classes without a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    CharUnits BaseOffset;
    CharUnits BaseOffsetFromNearestVBase;
    bool BaseDeclIsNonVirtualPrimaryBase;

    if (I.isVirtual()) {
      llvm_unreachable("NYI");
    } else {
      const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

      BaseOffset = Base.getBaseOffset() + Layout.getBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase =
          OffsetFromNearestVBase + Layout.getBaseClassOffset(BaseDecl);
      BaseDeclIsNonVirtualPrimaryBase = Layout.getPrimaryBase() == BaseDecl;
    }

    getVTablePointers(
        BaseSubobject(BaseDecl, BaseOffset),
        I.isVirtual() ? BaseDecl : NearestVBase, BaseOffsetFromNearestVBase,
        BaseDeclIsNonVirtualPrimaryBase, VTableClass, VBases, Vptrs);
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

  return Address(LoadCXXThis(), CXXThisAlignment);
}

void CIRGenFunction::buildInitializerForField(FieldDecl *Field, LValue LHS,
                                              Expr *Init) {
  QualType FieldType = Field->getType();
  switch (getEvaluationKind(FieldType)) {
  case TEK_Scalar:
    if (LHS.isSimple()) {
      buildExprAsInit(Init, Field, LHS, false);
    } else {
      llvm_unreachable("NYI");
    }
    break;
  case TEK_Complex:
    llvm_unreachable("NYI");
    break;
  case TEK_Aggregate: {
    AggValueSlot Slot = AggValueSlot::forLValue(
        LHS, AggValueSlot::IsDestructed, AggValueSlot::DoesNotNeedGCBarriers,
        AggValueSlot::IsNotAliased, getOverlapForFieldInit(Field),
        AggValueSlot::IsNotZeroed,
        // Checks are made by the code that calls constructor.
        AggValueSlot::IsSanitizerChecked);
    buildAggExpr(Init, Slot);
    break;
  }
  }

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

void CIRGenFunction::buildImplicitAssignmentOperatorBody(
    FunctionArgList &Args) {
  const CXXMethodDecl *AssignOp = cast<CXXMethodDecl>(CurGD.getDecl());
  const Stmt *RootS = AssignOp->getBody();
  assert(isa<CompoundStmt>(RootS) &&
         "Body of an implicit assignment operator should be compound stmt.");
  const CompoundStmt *RootCS = cast<CompoundStmt>(RootS);

  // LexicalScope Scope(*this, RootCS->getSourceRange());
  // FIXME(cir): add all of the below under a new scope.

  assert(!UnimplementedFeature::incrementProfileCounter());
  AssignmentMemcpyizer AM(*this, AssignOp, Args);
  for (auto *I : RootCS->body())
    AM.emitAssignment(I);
  AM.finish();
}

void CIRGenFunction::buildForwardingCallToLambda(
    const CXXMethodDecl *callOperator, CallArgList &callArgs) {
  // Get the address of the call operator.
  const auto &calleeFnInfo =
      CGM.getTypes().arrangeCXXMethodDeclaration(callOperator);
  auto calleePtr = CGM.GetAddrOfFunction(
      GlobalDecl(callOperator), CGM.getTypes().GetFunctionType(calleeFnInfo));

  // Prepare the return slot.
  const FunctionProtoType *FPT =
      callOperator->getType()->castAs<FunctionProtoType>();
  QualType resultType = FPT->getReturnType();
  ReturnValueSlot returnSlot;
  if (!resultType->isVoidType() &&
      calleeFnInfo.getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      !hasScalarEvaluationKind(calleeFnInfo.getReturnType())) {
    llvm_unreachable("NYI");
  }

  // We don't need to separately arrange the call arguments because
  // the call can't be variadic anyway --- it's impossible to forward
  // variadic arguments.

  // Now emit our call.
  auto callee = CIRGenCallee::forDirect(calleePtr, GlobalDecl(callOperator));
  RValue RV = buildCall(calleeFnInfo, callee, returnSlot, callArgs);

  // If necessary, copy the returned value into the slot.
  if (!resultType->isVoidType() && returnSlot.isNull()) {
    if (getLangOpts().ObjCAutoRefCount && resultType->isObjCRetainableType())
      llvm_unreachable("NYI");
    buildReturnOfRValue(*currSrcLoc, RV, resultType);
  } else {
    llvm_unreachable("NYI");
  }
}

void CIRGenFunction::buildLambdaDelegatingInvokeBody(const CXXMethodDecl *MD) {
  const CXXRecordDecl *Lambda = MD->getParent();

  // Start building arguments for forwarding call
  CallArgList CallArgs;

  QualType LambdaType = getContext().getRecordType(Lambda);
  QualType ThisType = getContext().getPointerType(LambdaType);
  Address ThisPtr =
      CreateMemTemp(LambdaType, getLoc(MD->getSourceRange()), "unused.capture");
  CallArgs.add(RValue::get(ThisPtr.getPointer()), ThisType);

  // Add the rest of the parameters.
  for (auto *Param : MD->parameters())
    buildDelegateCallArg(CallArgs, Param, Param->getBeginLoc());

  const CXXMethodDecl *CallOp = Lambda->getLambdaCallOperator();
  // For a generic lambda, find the corresponding call operator specialization
  // to which the call to the static-invoker shall be forwarded.
  if (Lambda->isGenericLambda()) {
    assert(MD->isFunctionTemplateSpecialization());
    const TemplateArgumentList *TAL = MD->getTemplateSpecializationArgs();
    FunctionTemplateDecl *CallOpTemplate =
        CallOp->getDescribedFunctionTemplate();
    void *InsertPos = nullptr;
    FunctionDecl *CorrespondingCallOpSpecialization =
        CallOpTemplate->findSpecialization(TAL->asArray(), InsertPos);
    assert(CorrespondingCallOpSpecialization);
    CallOp = cast<CXXMethodDecl>(CorrespondingCallOpSpecialization);
  }
  buildForwardingCallToLambda(CallOp, CallArgs);
}

void CIRGenFunction::buildLambdaStaticInvokeBody(const CXXMethodDecl *MD) {
  if (MD->isVariadic()) {
    // Codgen for LLVM doesn't emit code for this as well, it says:
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator
    // forward.
    llvm_unreachable("NYI");
  }

  buildLambdaDelegatingInvokeBody(MD);
}

void CIRGenFunction::destroyCXXObject(CIRGenFunction &CGF, Address addr,
                                      QualType type) {
  const RecordType *rtype = type->castAs<RecordType>();
  const CXXRecordDecl *record = cast<CXXRecordDecl>(rtype->getDecl());
  const CXXDestructorDecl *dtor = record->getDestructor();
  // TODO(cir): Unlike traditional codegen, CIRGen should actually emit trivial
  // dtors which shall be removed on later CIR passes. However, only remove this
  // assertion once we get a testcase to exercise this path.
  assert(!dtor->isTrivial());
  CGF.buildCXXDestructorCall(dtor, Dtor_Complete, /*for vbase*/ false,
                             /*Delegating=*/false, addr, type);
}

static bool FieldHasTrivialDestructorBody(ASTContext &Context,
                                          const FieldDecl *Field);

// FIXME(cir): this should be shared with traditional codegen.
static bool
HasTrivialDestructorBody(ASTContext &Context,
                         const CXXRecordDecl *BaseClassDecl,
                         const CXXRecordDecl *MostDerivedClassDecl) {
  // If the destructor is trivial we don't have to check anything else.
  if (BaseClassDecl->hasTrivialDestructor())
    return true;

  if (!BaseClassDecl->getDestructor()->hasTrivialBody())
    return false;

  // Check fields.
  for (const auto *Field : BaseClassDecl->fields())
    if (!FieldHasTrivialDestructorBody(Context, Field))
      return false;

  // Check non-virtual bases.
  for (const auto &I : BaseClassDecl->bases()) {
    if (I.isVirtual())
      continue;

    const CXXRecordDecl *NonVirtualBase =
        cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());
    if (!HasTrivialDestructorBody(Context, NonVirtualBase,
                                  MostDerivedClassDecl))
      return false;
  }

  if (BaseClassDecl == MostDerivedClassDecl) {
    // Check virtual bases.
    for (const auto &I : BaseClassDecl->vbases()) {
      const CXXRecordDecl *VirtualBase =
          cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());
      if (!HasTrivialDestructorBody(Context, VirtualBase, MostDerivedClassDecl))
        return false;
    }
  }

  return true;
}

// FIXME(cir): this should be shared with traditional codegen.
static bool FieldHasTrivialDestructorBody(ASTContext &Context,
                                          const FieldDecl *Field) {
  QualType FieldBaseElementType = Context.getBaseElementType(Field->getType());

  const RecordType *RT = FieldBaseElementType->getAs<RecordType>();
  if (!RT)
    return true;

  CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());

  // The destructor for an implicit anonymous union member is never invoked.
  if (FieldClassDecl->isUnion() && FieldClassDecl->isAnonymousStructOrUnion())
    return false;

  return HasTrivialDestructorBody(Context, FieldClassDecl, FieldClassDecl);
}

/// Check whether we need to initialize any vtable pointers before calling this
/// destructor.
/// FIXME(cir): this should be shared with traditional codegen.
static bool CanSkipVTablePointerInitialization(CIRGenFunction &CGF,
                                               const CXXDestructorDecl *Dtor) {
  const CXXRecordDecl *ClassDecl = Dtor->getParent();
  if (!ClassDecl->isDynamicClass())
    return true;

  // For a final class, the vtable pointer is known to already point to the
  // class's vtable.
  if (ClassDecl->isEffectivelyFinal())
    return true;

  if (!Dtor->hasTrivialBody())
    return false;

  // Check the fields.
  for (const auto *Field : ClassDecl->fields())
    if (!FieldHasTrivialDestructorBody(CGF.getContext(), Field))
      return false;

  return true;
}

/// Emits the body of the current destructor.
void CIRGenFunction::buildDestructorBody(FunctionArgList &Args) {
  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CurGD.getDecl());
  CXXDtorType DtorType = CurGD.getDtorType();

  // For an abstract class, non-base destructors are never used (and can't
  // be emitted in general, because vbase dtors may not have been validated
  // by Sema), but the Itanium ABI doesn't make them optional and Clang may
  // in fact emit references to them from other compilations, so emit them
  // as functions containing a trap instruction.
  if (DtorType != Dtor_Base && Dtor->getParent()->isAbstract()) {
    llvm_unreachable("NYI");
  }

  Stmt *Body = Dtor->getBody();
  if (Body)
    assert(!UnimplementedFeature::incrementProfileCounter());

  // The call to operator delete in a deleting destructor happens
  // outside of the function-try-block, which means it's always
  // possible to delegate the destructor body to the complete
  // destructor.  Do so.
  if (DtorType == Dtor_Deleting) {
    RunCleanupsScope DtorEpilogue(*this);
    EnterDtorCleanups(Dtor, Dtor_Deleting);
    if (HaveInsertPoint()) {
      QualType ThisTy = Dtor->getFunctionObjectParameterType();
      buildCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                             /*Delegating=*/false, LoadCXXThisAddress(),
                             ThisTy);
    }
    return;
  }

  // If the body is a function-try-block, enter the try before
  // anything else.
  bool isTryBody = (Body && isa<CXXTryStmt>(Body));
  if (isTryBody) {
    llvm_unreachable("NYI");
    // EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);
  }
  if (UnimplementedFeature::emitAsanPrologueOrEpilogue())
    llvm_unreachable("NYI");

  // Enter the epilogue cleanups.
  RunCleanupsScope DtorEpilogue(*this);

  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.  In the Microsoft ABI, we
  // always delegate because we might not have a definition in this TU.
  switch (DtorType) {
  case Dtor_Comdat:
    llvm_unreachable("not expecting a COMDAT");
  case Dtor_Deleting:
    llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    assert((Body || getTarget().getCXXABI().isMicrosoft()) &&
           "can't emit a dtor without a body for non-Microsoft ABIs");

    // Enter the cleanup scopes for virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Complete);

    if (!isTryBody) {
      QualType ThisTy = Dtor->getFunctionObjectParameterType();
      buildCXXDestructorCall(Dtor, Dtor_Base, /*ForVirtualBase=*/false,
                             /*Delegating=*/false, LoadCXXThisAddress(),
                             ThisTy);
      break;
    }

    // Fallthrough: act like we're in the base variant.
    [[fallthrough]];

  case Dtor_Base:
    assert(Body);

    // Enter the cleanup scopes for fields and non-virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Base);

    // Initialize the vtable pointers before entering the body.
    if (!CanSkipVTablePointerInitialization(*this, Dtor)) {
      // Insert the llvm.launder.invariant.group intrinsic before initializing
      // the vptrs to cancel any previous assumptions we might have made.
      if (CGM.getCodeGenOpts().StrictVTablePointers &&
          CGM.getCodeGenOpts().OptimizationLevel > 0)
        llvm_unreachable("NYI");
      llvm_unreachable("NYI");
    }

    if (isTryBody)
      llvm_unreachable("NYI");
    else if (Body)
      (void)buildStmt(Body, /*useCurrentScope=*/true);
    else {
      assert(Dtor->isImplicit() && "bodyless dtor not implicit");
      // nothing to do besides what's in the epilogue
    }
    // -fapple-kext must inline any call to this dtor into
    // the caller's body.
    if (getLangOpts().AppleKext)
      llvm_unreachable("NYI");

    break;
  }

  // Jump out through the epilogue cleanups.
  DtorEpilogue.ForceCleanup();

  // Exit the try if applicable.
  if (isTryBody)
    llvm_unreachable("NYI");
}

namespace {
[[maybe_unused]] mlir::Value
LoadThisForDtorDelete(CIRGenFunction &CGF, const CXXDestructorDecl *DD) {
  if (Expr *ThisArg = DD->getOperatorDeleteThisArg())
    return CGF.buildScalarExpr(ThisArg);
  return CGF.LoadCXXThis();
}

/// Call the operator delete associated with the current destructor.
struct CallDtorDelete final : EHScopeStack::Cleanup {
  CallDtorDelete() {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CGF.CurCodeDecl);
    const CXXRecordDecl *ClassDecl = Dtor->getParent();
    CGF.buildDeleteCall(Dtor->getOperatorDelete(),
                        LoadThisForDtorDelete(CGF, Dtor),
                        CGF.getContext().getTagDeclType(ClassDecl));
  }
};
} // namespace

/// Emit all code that comes at the end of class's destructor. This is to call
/// destructors on members and base classes in reverse order of their
/// construction.
///
/// For a deleting destructor, this also handles the case where a destroying
/// operator delete completely overrides the definition.
void CIRGenFunction::EnterDtorCleanups(const CXXDestructorDecl *DD,
                                       CXXDtorType DtorType) {
  assert((!DD->isTrivial() || DD->hasAttr<DLLExportAttr>()) &&
         "Should not emit dtor epilogue for non-exported trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() &&
           "operator delete missing - EnterDtorCleanups");
    if (CXXStructorImplicitParamValue) {
      llvm_unreachable("NYI");
    } else {
      if (DD->getOperatorDelete()->isDestroyingOperatorDelete()) {
        llvm_unreachable("NYI");
      } else {
        EHStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
      }
    }
    return;
  }

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // Unions have no bases and do not call field destructors.
  if (ClassDecl->isUnion())
    return;

  // The complete-destructor phase just destructs all the virtual bases.
  if (DtorType == Dtor_Complete) {
    // Poison the vtable pointer such that access after the base
    // and member destructors are invoked is invalid.
    if (CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
        SanOpts.has(SanitizerKind::Memory) && ClassDecl->getNumVBases() &&
        ClassDecl->isPolymorphic())
      assert(!UnimplementedFeature::sanitizeDtor());

    // We push them in the forward order so that they'll be popped in
    // the reverse order.
    for (const auto &Base : ClassDecl->vbases()) {
      auto *BaseClassDecl =
          cast<CXXRecordDecl>(Base.getType()->castAs<RecordType>()->getDecl());

      if (BaseClassDecl->hasTrivialDestructor()) {
        // Under SanitizeMemoryUseAfterDtor, poison the trivial base class
        // memory. For non-trival base classes the same is done in the class
        // destructor.
        assert(!UnimplementedFeature::sanitizeDtor());
      } else {
        EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, BaseClassDecl,
                                          /*BaseIsVirtual*/ true);
      }
    }

    return;
  }

  assert(DtorType == Dtor_Base);
  // Poison the vtable pointer if it has no virtual bases, but inherits
  // virtual functions.
  if (CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
      SanOpts.has(SanitizerKind::Memory) && !ClassDecl->getNumVBases() &&
      ClassDecl->isPolymorphic())
    assert(!UnimplementedFeature::sanitizeDtor());

  // Destroy non-virtual bases.
  for (const auto &Base : ClassDecl->bases()) {
    // Ignore virtual bases.
    if (Base.isVirtual())
      continue;

    CXXRecordDecl *BaseClassDecl = Base.getType()->getAsCXXRecordDecl();

    if (BaseClassDecl->hasTrivialDestructor()) {
      if (CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
          SanOpts.has(SanitizerKind::Memory) && !BaseClassDecl->isEmpty())
        assert(!UnimplementedFeature::sanitizeDtor());
    } else {
      EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, BaseClassDecl,
                                        /*BaseIsVirtual*/ false);
    }
  }

  // Poison fields such that access after their destructors are
  // invoked, and before the base class destructor runs, is invalid.
  bool SanitizeFields = CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
                        SanOpts.has(SanitizerKind::Memory);
  assert(!UnimplementedFeature::sanitizeDtor());

  // Destroy direct fields.
  for (const auto *Field : ClassDecl->fields()) {
    if (SanitizeFields)
      assert(!UnimplementedFeature::sanitizeDtor());

    QualType type = Field->getType();
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind)
      continue;

    // Anonymous union members do not have their destructors called.
    const RecordType *RT = type->getAsUnionType();
    if (RT && RT->getDecl()->isAnonymousStructOrUnion())
      continue;

    [[maybe_unused]] CleanupKind cleanupKind = getCleanupKind(dtorKind);
    llvm_unreachable("EHStack.pushCleanup<DestroyField>(...) NYI");
  }

  if (SanitizeFields)
    assert(!UnimplementedFeature::sanitizeDtor());
}

void CIRGenFunction::buildCXXDestructorCall(const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            bool ForVirtualBase,
                                            bool Delegating, Address This,
                                            QualType ThisTy) {
  CGM.getCXXABI().buildDestructorCall(*this, DD, Type, ForVirtualBase,
                                      Delegating, This, ThisTy);
}

mlir::Value CIRGenFunction::GetVTTParameter(GlobalDecl GD, bool ForVirtualBase,
                                            bool Delegating) {
  if (!CGM.getCXXABI().NeedsVTTParameter(GD)) {
    // This constructor/destructor does not need a VTT parameter.
    return nullptr;
  }

  const CXXRecordDecl *RD = cast<CXXMethodDecl>(CurCodeDecl)->getParent();
  const CXXRecordDecl *Base = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  if (Delegating) {
    llvm_unreachable("NYI");
  } else if (RD == Base) {
    llvm_unreachable("NYI");
  } else {
    llvm_unreachable("NYI");
  }

  if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
    llvm_unreachable("NYI");
  } else {
    llvm_unreachable("NYI");
  }
}

Address
CIRGenFunction::getAddressOfBaseClass(Address Value,
                                      const CXXRecordDecl *Derived,
                                      CastExpr::path_const_iterator PathBegin,
                                      CastExpr::path_const_iterator PathEnd,
                                      bool NullCheckValue, SourceLocation Loc) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  CastExpr::path_const_iterator Start = PathBegin;
  const CXXRecordDecl *VBase = nullptr;

  // Sema has done some convenient canonicalization here: if the
  // access path involved any virtual steps, the conversion path will
  // *start* with a step down to the correct virtual base subobject,
  // and hence will not require any further steps.
  if ((*Start)->isVirtual()) {
    llvm_unreachable("NYI");
  }

  // Compute the static offset of the ultimate destination within its
  // allocating subobject (the virtual base, if there is one, or else
  // the "complete" object that we see).
  CharUnits NonVirtualOffset = CGM.computeNonVirtualBaseClassOffset(
      VBase ? VBase : Derived, Start, PathEnd);

  // If there's a virtual step, we can sometimes "devirtualize" it.
  // For now, that's limited to when the derived type is final.
  // TODO: "devirtualize" this for accesses to known-complete objects.
  if (VBase && Derived->hasAttr<FinalAttr>()) {
    llvm_unreachable("NYI");
  }

  // Get the base pointer type.
  auto BaseValueTy = convertType((PathEnd[-1])->getType());
  assert(!UnimplementedFeature::addressSpace());
  // auto BasePtrTy = builder.getPointerTo(BaseValueTy);
  // QualType DerivedTy = getContext().getRecordType(Derived);
  // CharUnits DerivedAlign = CGM.getClassPointerAlignment(Derived);

  // If the static offset is zero and we don't have a virtual step,
  // just do a bitcast; null checks are unnecessary.
  if (NonVirtualOffset.isZero() && !VBase) {
    if (sanitizePerformTypeCheck()) {
      llvm_unreachable("NYI");
    }
    return builder.createBaseClassAddr(getLoc(Loc), Value, BaseValueTy);
  }

  // Skip over the offset (and the vtable load) if we're supposed to
  // null-check the pointer.
  if (NullCheckValue) {
    llvm_unreachable("NYI");
  }

  if (sanitizePerformTypeCheck()) {
    llvm_unreachable("NYI");
  }

  // Compute the virtual offset.
  mlir::Value VirtualOffset{};
  if (VBase) {
    llvm_unreachable("NYI");
  }

  // Apply both offsets.
  Value = ApplyNonVirtualAndVirtualOffset(*this, Value, NonVirtualOffset,
                                          VirtualOffset, Derived, VBase);
  // Cast to the destination type.
  Value = builder.createElementBitCast(Value.getPointer().getLoc(), Value,
                                       BaseValueTy);

  // Build a phi if we needed a null check.
  if (NullCheckValue) {
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
  return Value;
}

// TODO(cir): this can be shared with LLVM codegen.
bool CIRGenFunction::shouldEmitVTableTypeCheckedLoad(const CXXRecordDecl *RD) {
  if (!CGM.getCodeGenOpts().WholeProgramVTables ||
      !CGM.HasHiddenLTOVisibility(RD))
    return false;

  if (CGM.getCodeGenOpts().VirtualFunctionElimination)
    return true;

  if (!SanOpts.has(SanitizerKind::CFIVCall) ||
      !CGM.getCodeGenOpts().SanitizeTrap.has(SanitizerKind::CFIVCall))
    return false;

  std::string TypeName = RD->getQualifiedNameAsString();
  return !getContext().getNoSanitizeList().containsType(SanitizerKind::CFIVCall,
                                                        TypeName);
}

void CIRGenFunction::buildTypeMetadataCodeForVCall(const CXXRecordDecl *RD,
                                                   mlir::Value VTable,
                                                   SourceLocation Loc) {
  if (SanOpts.has(SanitizerKind::CFIVCall)) {
    llvm_unreachable("NYI");
  } else if (CGM.getCodeGenOpts().WholeProgramVTables &&
             // Don't insert type test assumes if we are forcing public
             // visibility.
             !CGM.AlwaysHasLTOVisibilityPublic(RD)) {
    llvm_unreachable("NYI");
  }
}

mlir::Value CIRGenFunction::getVTablePtr(mlir::Location Loc, Address This,
                                         mlir::Type VTableTy,
                                         const CXXRecordDecl *RD) {
  Address VTablePtrSrc = builder.createElementBitCast(Loc, This, VTableTy);
  auto VTable = builder.createLoad(Loc, VTablePtrSrc);
  assert(!UnimplementedFeature::tbaa());

  if (CGM.getCodeGenOpts().OptimizationLevel > 0 &&
      CGM.getCodeGenOpts().StrictVTablePointers) {
    assert(!UnimplementedFeature::createInvariantGroup());
  }

  return VTable;
}

Address CIRGenFunction::buildCXXMemberDataPointerAddress(
    const Expr *E, Address base, mlir::Value memberPtr,
    const MemberPointerType *memberPtrType, LValueBaseInfo *baseInfo) {
  assert(!UnimplementedFeature::cxxABI());

  auto op = builder.createGetIndirectMember(getLoc(E->getSourceRange()),
                                            base.getPointer(), memberPtr);

  QualType memberType = memberPtrType->getPointeeType();
  CharUnits memberAlign = CGM.getNaturalTypeAlignment(memberType, baseInfo);
  memberAlign = CGM.getDynamicOffsetAlignment(
      base.getAlignment(), memberPtrType->getClass()->getAsCXXRecordDecl(),
      memberAlign);

  return Address(op, convertTypeForMem(memberPtrType->getPointeeType()),
                 memberAlign);
}

clang::CharUnits
CIRGenModule::getDynamicOffsetAlignment(clang::CharUnits actualBaseAlign,
                                        const clang::CXXRecordDecl *baseDecl,
                                        clang::CharUnits expectedTargetAlign) {
  // If the base is an incomplete type (which is, alas, possible with
  // member pointers), be pessimistic.
  if (!baseDecl->isCompleteDefinition())
    return std::min(actualBaseAlign, expectedTargetAlign);

  auto &baseLayout = getASTContext().getASTRecordLayout(baseDecl);
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
  if (actualBaseAlign >= expectedBaseAlign) {
    return expectedTargetAlign;
  }

  // Otherwise, we might be offset by an arbitrary multiple of the
  // actual alignment.  The correct adjustment is to take the min of
  // the two alignments.
  return std::min(actualBaseAlign, expectedTargetAlign);
}
