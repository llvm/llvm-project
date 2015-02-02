//===--- CGClass.cpp - Emit LLVM Code for C++ classes ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CGBlocks.h"
#include "CGCXXABI.h"
#include "CGDebugInfo.h"
#include "CGRecordLayout.h"
#include "CodeGenFunction.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/CodeGenOptions.h"

using namespace clang;
using namespace CodeGen;

static CharUnits 
ComputeNonVirtualBaseClassOffset(ASTContext &Context, 
                                 const CXXRecordDecl *DerivedClass,
                                 CastExpr::path_const_iterator Start,
                                 CastExpr::path_const_iterator End) {
  CharUnits Offset = CharUnits::Zero();
  
  const CXXRecordDecl *RD = DerivedClass;
  
  for (CastExpr::path_const_iterator I = Start; I != End; ++I) {
    const CXXBaseSpecifier *Base = *I;
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    
    // Add the offset.
    Offset += Layout.getBaseClassOffset(BaseDecl);
    
    RD = BaseDecl;
  }
  
  return Offset;
}

llvm::Constant *
CodeGenModule::GetNonVirtualBaseClassOffset(const CXXRecordDecl *ClassDecl,
                                   CastExpr::path_const_iterator PathBegin,
                                   CastExpr::path_const_iterator PathEnd) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  CharUnits Offset = 
    ComputeNonVirtualBaseClassOffset(getContext(), ClassDecl,
                                     PathBegin, PathEnd);
  if (Offset.isZero())
    return nullptr;

  llvm::Type *PtrDiffTy = 
  Types.ConvertType(getContext().getPointerDiffType());
  
  return llvm::ConstantInt::get(PtrDiffTy, Offset.getQuantity());
}

/// Gets the address of a direct base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'This' is assumed to be non-null.
llvm::Value *
CodeGenFunction::GetAddressOfDirectBaseInCompleteClass(llvm::Value *This,
                                                   const CXXRecordDecl *Derived,
                                                   const CXXRecordDecl *Base,
                                                   bool BaseIsVirtual) {
  // 'this' must be a pointer (in some address space) to Derived.
  assert(This->getType()->isPointerTy() &&
         cast<llvm::PointerType>(This->getType())->getElementType()
           == ConvertType(Derived));

  // Compute the offset of the virtual base.
  CharUnits Offset;
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(Derived);
  if (BaseIsVirtual)
    Offset = Layout.getVBaseClassOffset(Base);
  else
    Offset = Layout.getBaseClassOffset(Base);

  // Shift and cast down to the base type.
  // TODO: for complete types, this should be possible with a GEP.
  llvm::Value *V = This;
  if (Offset.isPositive()) {
    V = Builder.CreateBitCast(V, Int8PtrTy);
    V = Builder.CreateConstInBoundsGEP1_64(V, Offset.getQuantity());
  }
  V = Builder.CreateBitCast(V, ConvertType(Base)->getPointerTo());

  return V;
}

static llvm::Value *
ApplyNonVirtualAndVirtualOffset(CodeGenFunction &CGF, llvm::Value *ptr,
                                CharUnits nonVirtualOffset,
                                llvm::Value *virtualOffset) {
  // Assert that we have something to do.
  assert(!nonVirtualOffset.isZero() || virtualOffset != nullptr);

  // Compute the offset from the static and dynamic components.
  llvm::Value *baseOffset;
  if (!nonVirtualOffset.isZero()) {
    baseOffset = llvm::ConstantInt::get(CGF.PtrDiffTy,
                                        nonVirtualOffset.getQuantity());
    if (virtualOffset) {
      baseOffset = CGF.Builder.CreateAdd(virtualOffset, baseOffset);
    }
  } else {
    baseOffset = virtualOffset;
  }
  
  // Apply the base offset.
  ptr = CGF.Builder.CreateBitCast(ptr, CGF.Int8PtrTy);
  ptr = CGF.Builder.CreateInBoundsGEP(ptr, baseOffset, "add.ptr");
  return ptr;
}

llvm::Value *CodeGenFunction::GetAddressOfBaseClass(
    llvm::Value *Value, const CXXRecordDecl *Derived,
    CastExpr::path_const_iterator PathBegin,
    CastExpr::path_const_iterator PathEnd, bool NullCheckValue,
    SourceLocation Loc) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  CastExpr::path_const_iterator Start = PathBegin;
  const CXXRecordDecl *VBase = nullptr;

  // Sema has done some convenient canonicalization here: if the
  // access path involved any virtual steps, the conversion path will
  // *start* with a step down to the correct virtual base subobject,
  // and hence will not require any further steps.
  if ((*Start)->isVirtual()) {
    VBase = 
      cast<CXXRecordDecl>((*Start)->getType()->getAs<RecordType>()->getDecl());
    ++Start;
  }

  // Compute the static offset of the ultimate destination within its
  // allocating subobject (the virtual base, if there is one, or else
  // the "complete" object that we see).
  CharUnits NonVirtualOffset = 
    ComputeNonVirtualBaseClassOffset(getContext(), VBase ? VBase : Derived,
                                     Start, PathEnd);

  // If there's a virtual step, we can sometimes "devirtualize" it.
  // For now, that's limited to when the derived type is final.
  // TODO: "devirtualize" this for accesses to known-complete objects.
  if (VBase && Derived->hasAttr<FinalAttr>()) {
    const ASTRecordLayout &layout = getContext().getASTRecordLayout(Derived);
    CharUnits vBaseOffset = layout.getVBaseClassOffset(VBase);
    NonVirtualOffset += vBaseOffset;
    VBase = nullptr; // we no longer have a virtual step
  }

  // Get the base pointer type.
  llvm::Type *BasePtrTy = 
    ConvertType((PathEnd[-1])->getType())->getPointerTo();

  QualType DerivedTy = getContext().getRecordType(Derived);
  CharUnits DerivedAlign = getContext().getTypeAlignInChars(DerivedTy);

  // If the static offset is zero and we don't have a virtual step,
  // just do a bitcast; null checks are unnecessary.
  if (NonVirtualOffset.isZero() && !VBase) {
    if (sanitizePerformTypeCheck()) {
      EmitTypeCheck(TCK_Upcast, Loc, Value, DerivedTy, DerivedAlign,
                    !NullCheckValue);
    }
    return Builder.CreateBitCast(Value, BasePtrTy);
  }

  llvm::BasicBlock *origBB = nullptr;
  llvm::BasicBlock *endBB = nullptr;

  // Skip over the offset (and the vtable load) if we're supposed to
  // null-check the pointer.
  if (NullCheckValue) {
    origBB = Builder.GetInsertBlock();
    llvm::BasicBlock *notNullBB = createBasicBlock("cast.notnull");
    endBB = createBasicBlock("cast.end");
    
    llvm::Value *isNull = Builder.CreateIsNull(Value);
    Builder.CreateCondBr(isNull, endBB, notNullBB);
    EmitBlock(notNullBB);
  }

  if (sanitizePerformTypeCheck()) {
    EmitTypeCheck(VBase ? TCK_UpcastToVirtualBase : TCK_Upcast, Loc, Value,
                  DerivedTy, DerivedAlign, true);
  }

  // Compute the virtual offset.
  llvm::Value *VirtualOffset = nullptr;
  if (VBase) {
    VirtualOffset =
      CGM.getCXXABI().GetVirtualBaseClassOffset(*this, Value, Derived, VBase);
  }

  // Apply both offsets.
  Value = ApplyNonVirtualAndVirtualOffset(*this, Value, 
                                          NonVirtualOffset,
                                          VirtualOffset);
  
  // Cast to the destination type.
  Value = Builder.CreateBitCast(Value, BasePtrTy);

  // Build a phi if we needed a null check.
  if (NullCheckValue) {
    llvm::BasicBlock *notNullBB = Builder.GetInsertBlock();
    Builder.CreateBr(endBB);
    EmitBlock(endBB);
    
    llvm::PHINode *PHI = Builder.CreatePHI(BasePtrTy, 2, "cast.result");
    PHI->addIncoming(Value, notNullBB);
    PHI->addIncoming(llvm::Constant::getNullValue(BasePtrTy), origBB);
    Value = PHI;
  }
  
  return Value;
}

llvm::Value *
CodeGenFunction::GetAddressOfDerivedClass(llvm::Value *Value,
                                          const CXXRecordDecl *Derived,
                                        CastExpr::path_const_iterator PathBegin,
                                          CastExpr::path_const_iterator PathEnd,
                                          bool NullCheckValue) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  QualType DerivedTy =
    getContext().getCanonicalType(getContext().getTagDeclType(Derived));
  llvm::Type *DerivedPtrTy = ConvertType(DerivedTy)->getPointerTo();

  llvm::Value *NonVirtualOffset =
    CGM.GetNonVirtualBaseClassOffset(Derived, PathBegin, PathEnd);
  
  if (!NonVirtualOffset) {
    // No offset, we can just cast back.
    return Builder.CreateBitCast(Value, DerivedPtrTy);
  }

  llvm::BasicBlock *CastNull = nullptr;
  llvm::BasicBlock *CastNotNull = nullptr;
  llvm::BasicBlock *CastEnd = nullptr;

  if (NullCheckValue) {
    CastNull = createBasicBlock("cast.null");
    CastNotNull = createBasicBlock("cast.notnull");
    CastEnd = createBasicBlock("cast.end");
    
    llvm::Value *IsNull = Builder.CreateIsNull(Value);
    Builder.CreateCondBr(IsNull, CastNull, CastNotNull);
    EmitBlock(CastNotNull);
  }
  
  // Apply the offset.
  Value = Builder.CreateBitCast(Value, Int8PtrTy);
  Value = Builder.CreateGEP(Value, Builder.CreateNeg(NonVirtualOffset),
                            "sub.ptr");

  // Just cast.
  Value = Builder.CreateBitCast(Value, DerivedPtrTy);

  if (NullCheckValue) {
    Builder.CreateBr(CastEnd);
    EmitBlock(CastNull);
    Builder.CreateBr(CastEnd);
    EmitBlock(CastEnd);
    
    llvm::PHINode *PHI = Builder.CreatePHI(Value->getType(), 2);
    PHI->addIncoming(Value, CastNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(Value->getType()), 
                     CastNull);
    Value = PHI;
  }
  
  return Value;
}

llvm::Value *CodeGenFunction::GetVTTParameter(GlobalDecl GD,
                                              bool ForVirtualBase,
                                              bool Delegating) {
  if (!CGM.getCXXABI().NeedsVTTParameter(GD)) {
    // This constructor/destructor does not need a VTT parameter.
    return nullptr;
  }
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(CurCodeDecl)->getParent();
  const CXXRecordDecl *Base = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  llvm::Value *VTT;

  uint64_t SubVTTIndex;

  if (Delegating) {
    // If this is a delegating constructor call, just load the VTT.
    return LoadCXXVTT();
  } else if (RD == Base) {
    // If the record matches the base, this is the complete ctor/dtor
    // variant calling the base variant in a class with virtual bases.
    assert(!CGM.getCXXABI().NeedsVTTParameter(CurGD) &&
           "doing no-op VTT offset in base dtor/ctor?");
    assert(!ForVirtualBase && "Can't have same class as virtual base!");
    SubVTTIndex = 0;
  } else {
    const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
    CharUnits BaseOffset = ForVirtualBase ? 
      Layout.getVBaseClassOffset(Base) : 
      Layout.getBaseClassOffset(Base);

    SubVTTIndex = 
      CGM.getVTables().getSubVTTIndex(RD, BaseSubobject(Base, BaseOffset));
    assert(SubVTTIndex != 0 && "Sub-VTT index must be greater than zero!");
  }
  
  if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
    // A VTT parameter was passed to the constructor, use it.
    VTT = LoadCXXVTT();
    VTT = Builder.CreateConstInBoundsGEP1_64(VTT, SubVTTIndex);
  } else {
    // We're the complete constructor, so get the VTT by name.
    VTT = CGM.getVTables().GetAddrOfVTT(RD);
    VTT = Builder.CreateConstInBoundsGEP2_64(VTT, 0, SubVTTIndex);
  }

  return VTT;
}

namespace {
  /// Call the destructor for a direct base class.
  struct CallBaseDtor : EHScopeStack::Cleanup {
    const CXXRecordDecl *BaseClass;
    bool BaseIsVirtual;
    CallBaseDtor(const CXXRecordDecl *Base, bool BaseIsVirtual)
      : BaseClass(Base), BaseIsVirtual(BaseIsVirtual) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      const CXXRecordDecl *DerivedClass =
        cast<CXXMethodDecl>(CGF.CurCodeDecl)->getParent();

      const CXXDestructorDecl *D = BaseClass->getDestructor();
      llvm::Value *Addr = 
        CGF.GetAddressOfDirectBaseInCompleteClass(CGF.LoadCXXThis(),
                                                  DerivedClass, BaseClass,
                                                  BaseIsVirtual);
      CGF.EmitCXXDestructorCall(D, Dtor_Base, BaseIsVirtual,
                                /*Delegating=*/false, Addr);
    }
  };

  /// A visitor which checks whether an initializer uses 'this' in a
  /// way which requires the vtable to be properly set.
  struct DynamicThisUseChecker : EvaluatedExprVisitor<DynamicThisUseChecker> {
    typedef EvaluatedExprVisitor<DynamicThisUseChecker> super;

    bool UsesThis;

    DynamicThisUseChecker(ASTContext &C) : super(C), UsesThis(false) {}

    // Black-list all explicit and implicit references to 'this'.
    //
    // Do we need to worry about external references to 'this' derived
    // from arbitrary code?  If so, then anything which runs arbitrary
    // external code might potentially access the vtable.
    void VisitCXXThisExpr(CXXThisExpr *E) { UsesThis = true; }
  };
}

static bool BaseInitializerUsesThis(ASTContext &C, const Expr *Init) {
  DynamicThisUseChecker Checker(C);
  Checker.Visit(const_cast<Expr*>(Init));
  return Checker.UsesThis;
}

static void EmitBaseInitializer(CodeGenFunction &CGF, 
                                const CXXRecordDecl *ClassDecl,
                                CXXCtorInitializer *BaseInit,
                                CXXCtorType CtorType) {
  assert(BaseInit->isBaseInitializer() &&
         "Must have base initializer!");

  llvm::Value *ThisPtr = CGF.LoadCXXThis();
  
  const Type *BaseType = BaseInit->getBaseClass();
  CXXRecordDecl *BaseClassDecl =
    cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());

  bool isBaseVirtual = BaseInit->isBaseVirtual();

  // The base constructor doesn't construct virtual bases.
  if (CtorType == Ctor_Base && isBaseVirtual)
    return;

  // If the initializer for the base (other than the constructor
  // itself) accesses 'this' in any way, we need to initialize the
  // vtables.
  if (BaseInitializerUsesThis(CGF.getContext(), BaseInit->getInit()))
    CGF.InitializeVTablePointers(ClassDecl);

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  llvm::Value *V = 
    CGF.GetAddressOfDirectBaseInCompleteClass(ThisPtr, ClassDecl,
                                              BaseClassDecl,
                                              isBaseVirtual);
  CharUnits Alignment = CGF.getContext().getTypeAlignInChars(BaseType);
  AggValueSlot AggSlot =
    AggValueSlot::forAddr(V, Alignment, Qualifiers(),
                          AggValueSlot::IsDestructed,
                          AggValueSlot::DoesNotNeedGCBarriers,
                          AggValueSlot::IsNotAliased);

  CGF.EmitAggExpr(BaseInit->getInit(), AggSlot);
  
  if (CGF.CGM.getLangOpts().Exceptions && 
      !BaseClassDecl->hasTrivialDestructor())
    CGF.EHStack.pushCleanup<CallBaseDtor>(EHCleanup, BaseClassDecl,
                                          isBaseVirtual);
}

static void EmitAggMemberInitializer(CodeGenFunction &CGF,
                                     LValue LHS,
                                     Expr *Init,
                                     llvm::Value *ArrayIndexVar,
                                     QualType T,
                                     ArrayRef<VarDecl *> ArrayIndexes,
                                     unsigned Index) {
  if (Index == ArrayIndexes.size()) {
    LValue LV = LHS;

    if (ArrayIndexVar) {
      // If we have an array index variable, load it and use it as an offset.
      // Then, increment the value.
      llvm::Value *Dest = LHS.getAddress();
      llvm::Value *ArrayIndex = CGF.Builder.CreateLoad(ArrayIndexVar);
      Dest = CGF.Builder.CreateInBoundsGEP(Dest, ArrayIndex, "destaddress");
      llvm::Value *Next = llvm::ConstantInt::get(ArrayIndex->getType(), 1);
      Next = CGF.Builder.CreateAdd(ArrayIndex, Next, "inc");
      CGF.Builder.CreateStore(Next, ArrayIndexVar);

      // Update the LValue.
      LV.setAddress(Dest);
      CharUnits Align = CGF.getContext().getTypeAlignInChars(T);
      LV.setAlignment(std::min(Align, LV.getAlignment()));
    }

    switch (CGF.getEvaluationKind(T)) {
    case TEK_Scalar:
      CGF.EmitScalarInit(Init, /*decl*/ nullptr, LV, false);
      break;
    case TEK_Complex:
      CGF.EmitComplexExprIntoLValue(Init, LV, /*isInit*/ true);
      break;
    case TEK_Aggregate: {
      AggValueSlot Slot =
        AggValueSlot::forLValue(LV,
                                AggValueSlot::IsDestructed,
                                AggValueSlot::DoesNotNeedGCBarriers,
                                AggValueSlot::IsNotAliased);

      CGF.EmitAggExpr(Init, Slot);
      break;
    }
    }

    return;
  }

  const ConstantArrayType *Array = CGF.getContext().getAsConstantArrayType(T);
  assert(Array && "Array initialization without the array type?");
  llvm::Value *IndexVar
    = CGF.GetAddrOfLocalVar(ArrayIndexes[Index]);
  assert(IndexVar && "Array index variable not loaded");
  
  // Initialize this index variable to zero.
  llvm::Value* Zero
    = llvm::Constant::getNullValue(
                              CGF.ConvertType(CGF.getContext().getSizeType()));
  CGF.Builder.CreateStore(Zero, IndexVar);
                                   
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = CGF.createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = CGF.createBasicBlock("for.end");
  
  CGF.EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = CGF.createBasicBlock("for.body");
  // Generate: if (loop-index < number-of-elements) fall to the loop body,
  // otherwise, go to the block after the for-loop.
  uint64_t NumElements = Array->getSize().getZExtValue();
  llvm::Value *Counter = CGF.Builder.CreateLoad(IndexVar);
  llvm::Value *NumElementsPtr =
    llvm::ConstantInt::get(Counter->getType(), NumElements);
  llvm::Value *IsLess = CGF.Builder.CreateICmpULT(Counter, NumElementsPtr,
                                                  "isless");
                                   
  // If the condition is true, execute the body.
  CGF.Builder.CreateCondBr(IsLess, ForBody, AfterFor);

  CGF.EmitBlock(ForBody);
  llvm::BasicBlock *ContinueBlock = CGF.createBasicBlock("for.inc");

  // Inside the loop body recurse to emit the inner loop or, eventually, the
  // constructor call.
  EmitAggMemberInitializer(CGF, LHS, Init, ArrayIndexVar,
                           Array->getElementType(), ArrayIndexes, Index + 1);

  CGF.EmitBlock(ContinueBlock);

  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(Counter->getType(), 1);
  Counter = CGF.Builder.CreateLoad(IndexVar);
  NextVal = CGF.Builder.CreateAdd(Counter, NextVal, "inc");
  CGF.Builder.CreateStore(NextVal, IndexVar);

  // Finally, branch back up to the condition for the next iteration.
  CGF.EmitBranch(CondBlock);

  // Emit the fall-through block.
  CGF.EmitBlock(AfterFor, true);
}

static void EmitMemberInitializer(CodeGenFunction &CGF,
                                  const CXXRecordDecl *ClassDecl,
                                  CXXCtorInitializer *MemberInit,
                                  const CXXConstructorDecl *Constructor,
                                  FunctionArgList &Args) {
  ApplyDebugLocation Loc(CGF, MemberInit->getSourceLocation());
  assert(MemberInit->isAnyMemberInitializer() &&
         "Must have member initializer!");
  assert(MemberInit->getInit() && "Must have initializer!");
  
  // non-static data member initializers.
  FieldDecl *Field = MemberInit->getAnyMember();
  QualType FieldType = Field->getType();

  llvm::Value *ThisPtr = CGF.LoadCXXThis();
  QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
  LValue LHS = CGF.MakeNaturalAlignAddrLValue(ThisPtr, RecordTy);

  if (MemberInit->isIndirectMemberInitializer()) {
    // If we are initializing an anonymous union field, drill down to
    // the field.
    IndirectFieldDecl *IndirectField = MemberInit->getIndirectMember();
    for (const auto *I : IndirectField->chain())
      LHS = CGF.EmitLValueForFieldInitialization(LHS, cast<FieldDecl>(I));
    FieldType = MemberInit->getIndirectMember()->getAnonField()->getType();
  } else {
    LHS = CGF.EmitLValueForFieldInitialization(LHS, Field);
  }

  // Special case: if we are in a copy or move constructor, and we are copying
  // an array of PODs or classes with trivial copy constructors, ignore the
  // AST and perform the copy we know is equivalent.
  // FIXME: This is hacky at best... if we had a bit more explicit information
  // in the AST, we could generalize it more easily.
  const ConstantArrayType *Array
    = CGF.getContext().getAsConstantArrayType(FieldType);
  if (Array && Constructor->isDefaulted() &&
      Constructor->isCopyOrMoveConstructor()) {
    QualType BaseElementTy = CGF.getContext().getBaseElementType(Array);
    CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(MemberInit->getInit());
    if (BaseElementTy.isPODType(CGF.getContext()) ||
        (CE && CE->getConstructor()->isTrivial())) {
      unsigned SrcArgIndex =
          CGF.CGM.getCXXABI().getSrcArgforCopyCtor(Constructor, Args);
      llvm::Value *SrcPtr
        = CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(Args[SrcArgIndex]));
      LValue ThisRHSLV = CGF.MakeNaturalAlignAddrLValue(SrcPtr, RecordTy);
      LValue Src = CGF.EmitLValueForFieldInitialization(ThisRHSLV, Field);
      
      // Copy the aggregate.
      CGF.EmitAggregateCopy(LHS.getAddress(), Src.getAddress(), FieldType,
                            LHS.isVolatileQualified());
      return;
    }
  }

  ArrayRef<VarDecl *> ArrayIndexes;
  if (MemberInit->getNumArrayIndices())
    ArrayIndexes = MemberInit->getArrayIndexes();
  CGF.EmitInitializerForField(Field, LHS, MemberInit->getInit(), ArrayIndexes);
}

void CodeGenFunction::EmitInitializerForField(
    FieldDecl *Field, LValue LHS, Expr *Init,
    ArrayRef<VarDecl *> ArrayIndexes) {
  QualType FieldType = Field->getType();
  switch (getEvaluationKind(FieldType)) {
  case TEK_Scalar:
    if (LHS.isSimple()) {
      EmitExprAsInit(Init, Field, LHS, false);
    } else {
      RValue RHS = RValue::get(EmitScalarExpr(Init));
      EmitStoreThroughLValue(RHS, LHS);
    }
    break;
  case TEK_Complex:
    EmitComplexExprIntoLValue(Init, LHS, /*isInit*/ true);
    break;
  case TEK_Aggregate: {
    llvm::Value *ArrayIndexVar = nullptr;
    if (ArrayIndexes.size()) {
      llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
      
      // The LHS is a pointer to the first object we'll be constructing, as
      // a flat array.
      QualType BaseElementTy = getContext().getBaseElementType(FieldType);
      llvm::Type *BasePtr = ConvertType(BaseElementTy);
      BasePtr = llvm::PointerType::getUnqual(BasePtr);
      llvm::Value *BaseAddrPtr = Builder.CreateBitCast(LHS.getAddress(), 
                                                       BasePtr);
      LHS = MakeAddrLValue(BaseAddrPtr, BaseElementTy);
      
      // Create an array index that will be used to walk over all of the
      // objects we're constructing.
      ArrayIndexVar = CreateTempAlloca(SizeTy, "object.index");
      llvm::Value *Zero = llvm::Constant::getNullValue(SizeTy);
      Builder.CreateStore(Zero, ArrayIndexVar);
      
      
      // Emit the block variables for the array indices, if any.
      for (unsigned I = 0, N = ArrayIndexes.size(); I != N; ++I)
        EmitAutoVarDecl(*ArrayIndexes[I]);
    }
    
    EmitAggMemberInitializer(*this, LHS, Init, ArrayIndexVar, FieldType,
                             ArrayIndexes, 0);
  }
  }

  // Ensure that we destroy this object if an exception is thrown
  // later in the constructor.
  QualType::DestructionKind dtorKind = FieldType.isDestructedType();
  if (needsEHCleanup(dtorKind))
    pushEHDestroy(dtorKind, LHS.getAddress(), FieldType);
}

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delegation optimization, i.e.
/// emitting the complete constructor as a simple call to the base
/// constructor.
static bool IsConstructorDelegationValid(const CXXConstructorDecl *Ctor) {

  // Currently we disable the optimization for classes with virtual
  // bases because (1) the addresses of parameter variables need to be
  // consistent across all initializers but (2) the delegate function
  // call necessarily creates a second copy of the parameter variable.
  //
  // The limiting example (purely theoretical AFAIK):
  //   struct A { A(int &c) { c++; } };
  //   struct B : virtual A {
  //     B(int count) : A(count) { printf("%d\n", count); }
  //   };
  // ...although even this example could in principle be emitted as a
  // delegation since the address of the parameter doesn't escape.
  if (Ctor->getParent()->getNumVBases()) {
    // TODO: white-list trivial vbase initializers.  This case wouldn't
    // be subject to the restrictions below.

    // TODO: white-list cases where:
    //  - there are no non-reference parameters to the constructor
    //  - the initializers don't access any non-reference parameters
    //  - the initializers don't take the address of non-reference
    //    parameters
    //  - etc.
    // If we ever add any of the above cases, remember that:
    //  - function-try-blocks will always blacklist this optimization
    //  - we need to perform the constructor prologue and cleanup in
    //    EmitConstructorBody.

    return false;
  }

  // We also disable the optimization for variadic functions because
  // it's impossible to "re-pass" varargs.
  if (Ctor->getType()->getAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (Ctor->isDelegatingConstructor())
    return false;

  return true;
}

// Emit code in ctor (Prologue==true) or dtor (Prologue==false)
// to poison the extra field paddings inserted under
// -fsanitize-address-field-padding=1|2.
void CodeGenFunction::EmitAsanPrologueOrEpilogue(bool Prologue) {
  ASTContext &Context = getContext();
  const CXXRecordDecl *ClassDecl =
      Prologue ? cast<CXXConstructorDecl>(CurGD.getDecl())->getParent()
               : cast<CXXDestructorDecl>(CurGD.getDecl())->getParent();
  if (!ClassDecl->mayInsertExtraPadding()) return;

  struct SizeAndOffset {
    uint64_t Size;
    uint64_t Offset;
  };

  unsigned PtrSize = CGM.getDataLayout().getPointerSizeInBits();
  const ASTRecordLayout &Info = Context.getASTRecordLayout(ClassDecl);

  // Populate sizes and offsets of fields.
  SmallVector<SizeAndOffset, 16> SSV(Info.getFieldCount());
  for (unsigned i = 0, e = Info.getFieldCount(); i != e; ++i)
    SSV[i].Offset =
        Context.toCharUnitsFromBits(Info.getFieldOffset(i)).getQuantity();

  size_t NumFields = 0;
  for (const auto *Field : ClassDecl->fields()) {
    const FieldDecl *D = Field;
    std::pair<CharUnits, CharUnits> FieldInfo =
        Context.getTypeInfoInChars(D->getType());
    CharUnits FieldSize = FieldInfo.first;
    assert(NumFields < SSV.size());
    SSV[NumFields].Size = D->isBitField() ? 0 : FieldSize.getQuantity();
    NumFields++;
  }
  assert(NumFields == SSV.size());
  if (SSV.size() <= 1) return;

  // We will insert calls to __asan_* run-time functions.
  // LLVM AddressSanitizer pass may decide to inline them later.
  llvm::Type *Args[2] = {IntPtrTy, IntPtrTy};
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(CGM.VoidTy, Args, false);
  llvm::Constant *F = CGM.CreateRuntimeFunction(
      FTy, Prologue ? "__asan_poison_intra_object_redzone"
                    : "__asan_unpoison_intra_object_redzone");

  llvm::Value *ThisPtr = LoadCXXThis();
  ThisPtr = Builder.CreatePtrToInt(ThisPtr, IntPtrTy);
  uint64_t TypeSize = Info.getNonVirtualSize().getQuantity();
  // For each field check if it has sufficient padding,
  // if so (un)poison it with a call.
  for (size_t i = 0; i < SSV.size(); i++) {
    uint64_t AsanAlignment = 8;
    uint64_t NextField = i == SSV.size() - 1 ? TypeSize : SSV[i + 1].Offset;
    uint64_t PoisonSize = NextField - SSV[i].Offset - SSV[i].Size;
    uint64_t EndOffset = SSV[i].Offset + SSV[i].Size;
    if (PoisonSize < AsanAlignment || !SSV[i].Size ||
        (NextField % AsanAlignment) != 0)
      continue;
    Builder.CreateCall2(
        F, Builder.CreateAdd(ThisPtr, Builder.getIntN(PtrSize, EndOffset)),
        Builder.getIntN(PtrSize, PoisonSize));
  }
}

/// EmitConstructorBody - Emits the body of the current constructor.
void CodeGenFunction::EmitConstructorBody(FunctionArgList &Args) {
  EmitAsanPrologueOrEpilogue(true);
  const CXXConstructorDecl *Ctor = cast<CXXConstructorDecl>(CurGD.getDecl());
  CXXCtorType CtorType = CurGD.getCtorType();

  assert((CGM.getTarget().getCXXABI().hasConstructorVariants() ||
          CtorType == Ctor_Complete) &&
         "can only generate complete ctor for this ABI");

  // Before we go any further, try the complete->base constructor
  // delegation optimization.
  if (CtorType == Ctor_Complete && IsConstructorDelegationValid(Ctor) &&
      CGM.getTarget().getCXXABI().hasConstructorVariants()) {
    EmitDelegateCXXConstructorCall(Ctor, Ctor_Base, Args, Ctor->getLocEnd());
    return;
  }

  const FunctionDecl *Definition = 0;
  Stmt *Body = Ctor->getBody(Definition);
  assert(Definition == Ctor && "emitting wrong constructor body");

  // Enter the function-try-block before the constructor prologue if
  // applicable.
  bool IsTryBody = (Body && isa<CXXTryStmt>(Body));
  if (IsTryBody)
    EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);

  RegionCounter Cnt = getPGORegionCounter(Body);
  Cnt.beginRegion(Builder);

  RunCleanupsScope RunCleanups(*this);

  // TODO: in restricted cases, we can emit the vbase initializers of
  // a complete ctor and then delegate to the base ctor.

  // Emit the constructor prologue, i.e. the base and member
  // initializers.
  EmitCtorPrologue(Ctor, CtorType, Args);

  // Emit the body of the statement.
  if (IsTryBody)
    EmitStmt(cast<CXXTryStmt>(Body)->getTryBlock());
  else if (Body)
    EmitStmt(Body);

  // Emit any cleanup blocks associated with the member or base
  // initializers, which includes (along the exceptional path) the
  // destructors for those members and bases that were fully
  // constructed.
  RunCleanups.ForceCleanup();

  if (IsTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), true);
}

namespace {
  /// RAII object to indicate that codegen is copying the value representation
  /// instead of the object representation. Useful when copying a struct or
  /// class which has uninitialized members and we're only performing
  /// lvalue-to-rvalue conversion on the object but not its members.
  class CopyingValueRepresentation {
  public:
    explicit CopyingValueRepresentation(CodeGenFunction &CGF)
        : CGF(CGF), OldSanOpts(CGF.SanOpts) {
      CGF.SanOpts.set(SanitizerKind::Bool, false);
      CGF.SanOpts.set(SanitizerKind::Enum, false);
    }
    ~CopyingValueRepresentation() {
      CGF.SanOpts = OldSanOpts;
    }
  private:
    CodeGenFunction &CGF;
    SanitizerSet OldSanOpts;
  };
}

namespace {
  class FieldMemcpyizer {
  public:
    FieldMemcpyizer(CodeGenFunction &CGF, const CXXRecordDecl *ClassDecl,
                    const VarDecl *SrcRec)
      : CGF(CGF), ClassDecl(ClassDecl), SrcRec(SrcRec), 
        RecLayout(CGF.getContext().getASTRecordLayout(ClassDecl)),
        FirstField(nullptr), LastField(nullptr), FirstFieldOffset(0),
        LastFieldOffset(0), LastAddedFieldIndex(0) {}

    bool isMemcpyableField(FieldDecl *F) const {
      // Never memcpy fields when we are adding poisoned paddings.
      if (CGF.getContext().getLangOpts().SanitizeAddressFieldPadding)
        return false;
      Qualifiers Qual = F->getType().getQualifiers();
      if (Qual.hasVolatile() || Qual.hasObjCLifetime())
        return false;
      return true;
    }

    void addMemcpyableField(FieldDecl *F) {
      if (!FirstField)
        addInitialField(F);
      else
        addNextField(F);
    }

    CharUnits getMemcpySize(uint64_t FirstByteOffset) const {
      unsigned LastFieldSize =
        LastField->isBitField() ?
          LastField->getBitWidthValue(CGF.getContext()) :
          CGF.getContext().getTypeSize(LastField->getType()); 
      uint64_t MemcpySizeBits =
        LastFieldOffset + LastFieldSize - FirstByteOffset +
        CGF.getContext().getCharWidth() - 1;
      CharUnits MemcpySize =
        CGF.getContext().toCharUnitsFromBits(MemcpySizeBits);
      return MemcpySize;
    }

    void emitMemcpy() {
      // Give the subclass a chance to bail out if it feels the memcpy isn't
      // worth it (e.g. Hasn't aggregated enough data).
      if (!FirstField) {
        return;
      }

      CharUnits Alignment;

      uint64_t FirstByteOffset;
      if (FirstField->isBitField()) {
        const CGRecordLayout &RL =
          CGF.getTypes().getCGRecordLayout(FirstField->getParent());
        const CGBitFieldInfo &BFInfo = RL.getBitFieldInfo(FirstField);
        Alignment = CharUnits::fromQuantity(BFInfo.StorageAlignment);
        // FirstFieldOffset is not appropriate for bitfields,
        // it won't tell us what the storage offset should be and thus might not
        // be properly aligned.
        //
        // Instead calculate the storage offset using the offset of the field in
        // the struct type.
        const llvm::DataLayout &DL = CGF.CGM.getDataLayout();
        FirstByteOffset =
            DL.getStructLayout(RL.getLLVMType())
                ->getElementOffsetInBits(RL.getLLVMFieldNo(FirstField));
      } else {
        Alignment = CGF.getContext().getDeclAlign(FirstField);
        FirstByteOffset = FirstFieldOffset;
      }

      assert((CGF.getContext().toCharUnitsFromBits(FirstByteOffset) %
              Alignment) == 0 && "Bad field alignment.");

      CharUnits MemcpySize = getMemcpySize(FirstByteOffset);
      QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
      llvm::Value *ThisPtr = CGF.LoadCXXThis();
      LValue DestLV = CGF.MakeNaturalAlignAddrLValue(ThisPtr, RecordTy);
      LValue Dest = CGF.EmitLValueForFieldInitialization(DestLV, FirstField);
      llvm::Value *SrcPtr = CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(SrcRec));
      LValue SrcLV = CGF.MakeNaturalAlignAddrLValue(SrcPtr, RecordTy);
      LValue Src = CGF.EmitLValueForFieldInitialization(SrcLV, FirstField);

      emitMemcpyIR(Dest.isBitField() ? Dest.getBitFieldAddr() : Dest.getAddress(),
                   Src.isBitField() ? Src.getBitFieldAddr() : Src.getAddress(),
                   MemcpySize, Alignment);
      reset();
    }

    void reset() {
      FirstField = nullptr;
    }

  protected:
    CodeGenFunction &CGF;
    const CXXRecordDecl *ClassDecl;

  private:

    void emitMemcpyIR(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                      CharUnits Size, CharUnits Alignment) {
      llvm::PointerType *DPT = cast<llvm::PointerType>(DestPtr->getType());
      llvm::Type *DBP =
        llvm::Type::getInt8PtrTy(CGF.getLLVMContext(), DPT->getAddressSpace());
      DestPtr = CGF.Builder.CreateBitCast(DestPtr, DBP);

      llvm::PointerType *SPT = cast<llvm::PointerType>(SrcPtr->getType());
      llvm::Type *SBP =
        llvm::Type::getInt8PtrTy(CGF.getLLVMContext(), SPT->getAddressSpace());
      SrcPtr = CGF.Builder.CreateBitCast(SrcPtr, SBP);

      CGF.Builder.CreateMemCpy(DestPtr, SrcPtr, Size.getQuantity(),
                               Alignment.getQuantity());
    }

    void addInitialField(FieldDecl *F) {
        FirstField = F;
        LastField = F;
        FirstFieldOffset = RecLayout.getFieldOffset(F->getFieldIndex());
        LastFieldOffset = FirstFieldOffset;
        LastAddedFieldIndex = F->getFieldIndex();
        return;
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
      } else if (FOffset > LastFieldOffset) {
        LastField = F;
        LastFieldOffset = FOffset;
      }
    }

    const VarDecl *SrcRec;
    const ASTRecordLayout &RecLayout;
    FieldDecl *FirstField;
    FieldDecl *LastField;
    uint64_t FirstFieldOffset, LastFieldOffset;
    unsigned LastAddedFieldIndex;
  };

  class ConstructorMemcpyizer : public FieldMemcpyizer {
  private:

    /// Get source argument for copy constructor. Returns null if not a copy
    /// constructor.
    static const VarDecl *getTrivialCopySource(CodeGenFunction &CGF,
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
      FieldDecl *Field = MemberInit->getMember();
      assert(Field && "No field for member init.");
      QualType FieldType = Field->getType();
      CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(MemberInit->getInit());

      // Bail out on non-POD, not-trivially-constructable members.
      if (!(CE && CE->getConstructor()->isTrivial()) &&
          !(FieldType.isTriviallyCopyableType(CGF.getContext()) ||
            FieldType->isReferenceType()))
        return false;

      // Bail out on volatile fields.
      if (!isMemcpyableField(Field))
        return false;

      // Otherwise we're good.
      return true;
    }

  public:
    ConstructorMemcpyizer(CodeGenFunction &CGF, const CXXConstructorDecl *CD,
                          FunctionArgList &Args)
      : FieldMemcpyizer(CGF, CD->getParent(), getTrivialCopySource(CGF, CD, Args)),
        ConstructorDecl(CD),
        MemcpyableCtor(CD->isDefaulted() &&
                       CD->isCopyOrMoveConstructor() &&
                       CGF.getLangOpts().getGC() == LangOptions::NonGC),
        Args(Args) { }

    void addMemberInitializer(CXXCtorInitializer *MemberInit) {
      if (isMemberInitMemcpyable(MemberInit)) {
        AggregatedInits.push_back(MemberInit);
        addMemcpyableField(MemberInit->getMember());
      } else {
        emitAggregatedInits();
        EmitMemberInitializer(CGF, ConstructorDecl->getParent(), MemberInit,
                              ConstructorDecl, Args);
      }
    }

    void emitAggregatedInits() {
      if (AggregatedInits.size() <= 1) {
        // This memcpy is too small to be worthwhile. Fall back on default
        // codegen.
        if (!AggregatedInits.empty()) {
          CopyingValueRepresentation CVR(CGF);
          EmitMemberInitializer(CGF, ConstructorDecl->getParent(),
                                AggregatedInits[0], ConstructorDecl, Args);
        }
        reset();
        return;
      }

      pushEHDestructors();
      emitMemcpy();
      AggregatedInits.clear();
    }

    void pushEHDestructors() {
      llvm::Value *ThisPtr = CGF.LoadCXXThis();
      QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
      LValue LHS = CGF.MakeNaturalAlignAddrLValue(ThisPtr, RecordTy);

      for (unsigned i = 0; i < AggregatedInits.size(); ++i) {
        QualType FieldType = AggregatedInits[i]->getMember()->getType();
        QualType::DestructionKind dtorKind = FieldType.isDestructedType();
        if (CGF.needsEHCleanup(dtorKind))
          CGF.pushEHDestroy(dtorKind, LHS.getAddress(), FieldType);
      }
    }

    void finish() {
      emitAggregatedInits();
    }

  private:
    const CXXConstructorDecl *ConstructorDecl;
    bool MemcpyableCtor;
    FunctionArgList &Args;
    SmallVector<CXXCtorInitializer*, 16> AggregatedInits;
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
        MemberExpr *ME2 = dyn_cast<MemberExpr>(RHS);
        if (dyn_cast<FieldDecl>(ME2->getMemberDecl()) != Field)
          return nullptr;
        return Field;
      } else if (CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(S)) {
        CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MCE->getCalleeDecl());
        if (!(MD && (MD->isCopyAssignmentOperator() ||
                       MD->isMoveAssignmentOperator()) &&
              MD->isTrivial()))
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
    SmallVector<Stmt*, 16> AggregatedStmts;

  public:

    AssignmentMemcpyizer(CodeGenFunction &CGF, const CXXMethodDecl *AD,
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
        CGF.EmitStmt(S);
      }
    }

    void emitAggregatedStmts() {
      if (AggregatedStmts.size() <= 1) {
        if (!AggregatedStmts.empty()) {
          CopyingValueRepresentation CVR(CGF);
          CGF.EmitStmt(AggregatedStmts[0]);
        }
        reset();
      }

      emitMemcpy();
      AggregatedStmts.clear();
    }

    void finish() {
      emitAggregatedStmts();
    }
  };

}

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD,
                                       CXXCtorType CtorType,
                                       FunctionArgList &Args) {
  if (CD->isDelegatingConstructor())
    return EmitDelegatingCXXConstructorCall(CD, Args);

  const CXXRecordDecl *ClassDecl = CD->getParent();

  CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
                                          E = CD->init_end();

  llvm::BasicBlock *BaseCtorContinueBB = nullptr;
  if (ClassDecl->getNumVBases() &&
      !CGM.getTarget().getCXXABI().hasConstructorVariants()) {
    // The ABIs that don't have constructor variants need to put a branch
    // before the virtual base initialization code.
    BaseCtorContinueBB =
      CGM.getCXXABI().EmitCtorCompleteObjectHandler(*this, ClassDecl);
    assert(BaseCtorContinueBB);
  }

  // Virtual base initializers first.
  for (; B != E && (*B)->isBaseInitializer() && (*B)->isBaseVirtual(); B++) {
    EmitBaseInitializer(*this, ClassDecl, *B, CtorType);
  }

  if (BaseCtorContinueBB) {
    // Complete object handler should continue to the remaining initializers.
    Builder.CreateBr(BaseCtorContinueBB);
    EmitBlock(BaseCtorContinueBB);
  }

  // Then, non-virtual base initializers.
  for (; B != E && (*B)->isBaseInitializer(); B++) {
    assert(!(*B)->isBaseVirtual());
    EmitBaseInitializer(*this, ClassDecl, *B, CtorType);
  }

  InitializeVTablePointers(ClassDecl);

  // And finally, initialize class members.
  FieldConstructionScope FCS(*this, CXXThisValue);
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

static bool
FieldHasTrivialDestructorBody(ASTContext &Context, const FieldDecl *Field);

static bool
HasTrivialDestructorBody(ASTContext &Context, 
                         const CXXRecordDecl *BaseClassDecl,
                         const CXXRecordDecl *MostDerivedClassDecl)
{
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
      if (!HasTrivialDestructorBody(Context, VirtualBase,
                                    MostDerivedClassDecl))
        return false;      
    }
  }

  return true;
}

static bool
FieldHasTrivialDestructorBody(ASTContext &Context,
                              const FieldDecl *Field)
{
  QualType FieldBaseElementType = Context.getBaseElementType(Field->getType());

  const RecordType *RT = FieldBaseElementType->getAs<RecordType>();
  if (!RT)
    return true;
  
  CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
  return HasTrivialDestructorBody(Context, FieldClassDecl, FieldClassDecl);
}

/// CanSkipVTablePointerInitialization - Check whether we need to initialize
/// any vtable pointers before calling this destructor.
static bool CanSkipVTablePointerInitialization(ASTContext &Context,
                                               const CXXDestructorDecl *Dtor) {
  if (!Dtor->hasTrivialBody())
    return false;

  // Check the fields.
  const CXXRecordDecl *ClassDecl = Dtor->getParent();
  for (const auto *Field : ClassDecl->fields())
    if (!FieldHasTrivialDestructorBody(Context, Field))
      return false;

  return true;
}

/// EmitDestructorBody - Emits the body of the current destructor.
void CodeGenFunction::EmitDestructorBody(FunctionArgList &Args) {
  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CurGD.getDecl());
  CXXDtorType DtorType = CurGD.getDtorType();

  // The call to operator delete in a deleting destructor happens
  // outside of the function-try-block, which means it's always
  // possible to delegate the destructor body to the complete
  // destructor.  Do so.
  if (DtorType == Dtor_Deleting) {
    EnterDtorCleanups(Dtor, Dtor_Deleting);
    EmitCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                          /*Delegating=*/false, LoadCXXThis());
    PopCleanupBlock();
    return;
  }

  Stmt *Body = Dtor->getBody();

  // If the body is a function-try-block, enter the try before
  // anything else.
  bool isTryBody = (Body && isa<CXXTryStmt>(Body));
  if (isTryBody)
    EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);
  EmitAsanPrologueOrEpilogue(false);

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

  case Dtor_Deleting: llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    assert((Body || getTarget().getCXXABI().isMicrosoft()) &&
           "can't emit a dtor without a body for non-Microsoft ABIs");

    // Enter the cleanup scopes for virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Complete);

    if (!isTryBody) {
      EmitCXXDestructorCall(Dtor, Dtor_Base, /*ForVirtualBase=*/false,
                            /*Delegating=*/false, LoadCXXThis());
      break;
    }
    // Fallthrough: act like we're in the base variant.
      
  case Dtor_Base:
    assert(Body);

    RegionCounter Cnt = getPGORegionCounter(Body);
    Cnt.beginRegion(Builder);

    // Enter the cleanup scopes for fields and non-virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Base);

    // Initialize the vtable pointers before entering the body.
    if (!CanSkipVTablePointerInitialization(getContext(), Dtor))
        InitializeVTablePointers(Dtor->getParent());

    if (isTryBody)
      EmitStmt(cast<CXXTryStmt>(Body)->getTryBlock());
    else if (Body)
      EmitStmt(Body);
    else {
      assert(Dtor->isImplicit() && "bodyless dtor not implicit");
      // nothing to do besides what's in the epilogue
    }
    // -fapple-kext must inline any call to this dtor into
    // the caller's body.
    if (getLangOpts().AppleKext)
      CurFn->addFnAttr(llvm::Attribute::AlwaysInline);
    break;
  }

  // Jump out through the epilogue cleanups.
  DtorEpilogue.ForceCleanup();

  // Exit the try if applicable.
  if (isTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), true);
}

void CodeGenFunction::emitImplicitAssignmentOperatorBody(FunctionArgList &Args) {
  const CXXMethodDecl *AssignOp = cast<CXXMethodDecl>(CurGD.getDecl());
  const Stmt *RootS = AssignOp->getBody();
  assert(isa<CompoundStmt>(RootS) &&
         "Body of an implicit assignment operator should be compound stmt.");
  const CompoundStmt *RootCS = cast<CompoundStmt>(RootS);

  LexicalScope Scope(*this, RootCS->getSourceRange());

  AssignmentMemcpyizer AM(*this, AssignOp, Args);
  for (auto *I : RootCS->body())
    AM.emitAssignment(I);  
  AM.finish();
}

namespace {
  /// Call the operator delete associated with the current destructor.
  struct CallDtorDelete : EHScopeStack::Cleanup {
    CallDtorDelete() {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CGF.CurCodeDecl);
      const CXXRecordDecl *ClassDecl = Dtor->getParent();
      CGF.EmitDeleteCall(Dtor->getOperatorDelete(), CGF.LoadCXXThis(),
                         CGF.getContext().getTagDeclType(ClassDecl));
    }
  };

  struct CallDtorDeleteConditional : EHScopeStack::Cleanup {
    llvm::Value *ShouldDeleteCondition;
  public:
    CallDtorDeleteConditional(llvm::Value *ShouldDeleteCondition)
      : ShouldDeleteCondition(ShouldDeleteCondition) {
      assert(ShouldDeleteCondition != nullptr);
    }

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      llvm::BasicBlock *callDeleteBB = CGF.createBasicBlock("dtor.call_delete");
      llvm::BasicBlock *continueBB = CGF.createBasicBlock("dtor.continue");
      llvm::Value *ShouldCallDelete
        = CGF.Builder.CreateIsNull(ShouldDeleteCondition);
      CGF.Builder.CreateCondBr(ShouldCallDelete, continueBB, callDeleteBB);

      CGF.EmitBlock(callDeleteBB);
      const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CGF.CurCodeDecl);
      const CXXRecordDecl *ClassDecl = Dtor->getParent();
      CGF.EmitDeleteCall(Dtor->getOperatorDelete(), CGF.LoadCXXThis(),
                         CGF.getContext().getTagDeclType(ClassDecl));
      CGF.Builder.CreateBr(continueBB);

      CGF.EmitBlock(continueBB);
    }
  };

  class DestroyField  : public EHScopeStack::Cleanup {
    const FieldDecl *field;
    CodeGenFunction::Destroyer *destroyer;
    bool useEHCleanupForArray;

  public:
    DestroyField(const FieldDecl *field, CodeGenFunction::Destroyer *destroyer,
                 bool useEHCleanupForArray)
      : field(field), destroyer(destroyer),
        useEHCleanupForArray(useEHCleanupForArray) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      // Find the address of the field.
      llvm::Value *thisValue = CGF.LoadCXXThis();
      QualType RecordTy = CGF.getContext().getTagDeclType(field->getParent());
      LValue ThisLV = CGF.MakeAddrLValue(thisValue, RecordTy);
      LValue LV = CGF.EmitLValueForField(ThisLV, field);
      assert(LV.isSimple());
      
      CGF.emitDestroy(LV.getAddress(), field->getType(), destroyer,
                      flags.isForNormalCleanup() && useEHCleanupForArray);
    }
  };
}

/// \brief Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes
/// in reverse order of their construction.
void CodeGenFunction::EnterDtorCleanups(const CXXDestructorDecl *DD,
                                        CXXDtorType DtorType) {
  assert((!DD->isTrivial() || DD->hasAttr<DLLExportAttr>()) &&
         "Should not emit dtor epilogue for non-exported trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() && 
           "operator delete missing - EnterDtorCleanups");
    if (CXXStructorImplicitParamValue) {
      // If there is an implicit param to the deleting dtor, it's a boolean
      // telling whether we should call delete at the end of the dtor.
      EHStack.pushCleanup<CallDtorDeleteConditional>(
          NormalAndEHCleanup, CXXStructorImplicitParamValue);
    } else {
      EHStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
    }
    return;
  }

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // Unions have no bases and do not call field destructors.
  if (ClassDecl->isUnion())
    return;

  // The complete-destructor phase just destructs all the virtual bases.
  if (DtorType == Dtor_Complete) {

    // We push them in the forward order so that they'll be popped in
    // the reverse order.
    for (const auto &Base : ClassDecl->vbases()) {
      CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    
      // Ignore trivial destructors.
      if (BaseClassDecl->hasTrivialDestructor())
        continue;

      EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup,
                                        BaseClassDecl,
                                        /*BaseIsVirtual*/ true);
    }

    return;
  }

  assert(DtorType == Dtor_Base);
  
  // Destroy non-virtual bases.
  for (const auto &Base : ClassDecl->bases()) {
    // Ignore virtual bases.
    if (Base.isVirtual())
      continue;
    
    CXXRecordDecl *BaseClassDecl = Base.getType()->getAsCXXRecordDecl();
    
    // Ignore trivial destructors.
    if (BaseClassDecl->hasTrivialDestructor())
      continue;

    EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup,
                                      BaseClassDecl,
                                      /*BaseIsVirtual*/ false);
  }

  // Destroy direct fields.
  for (const auto *Field : ClassDecl->fields()) {
    QualType type = Field->getType();
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind) continue;

    // Anonymous union members do not have their destructors called.
    const RecordType *RT = type->getAsUnionType();
    if (RT && RT->getDecl()->isAnonymousStructOrUnion()) continue;

    CleanupKind cleanupKind = getCleanupKind(dtorKind);
    EHStack.pushCleanup<DestroyField>(cleanupKind, Field,
                                      getDestroyer(dtorKind),
                                      cleanupKind & EHCleanup);
  }
}

/// EmitCXXAggrConstructorCall - Emit a loop to call a particular
/// constructor for each of several members of an array.
///
/// \param ctor the constructor to call for each element
/// \param arrayType the type of the array to initialize
/// \param arrayBegin an arrayType*
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CodeGenFunction::EmitCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, const ConstantArrayType *arrayType,
    llvm::Value *arrayBegin, const CXXConstructExpr *E, bool zeroInitialize) {
  QualType elementType;
  llvm::Value *numElements =
    emitArrayLength(arrayType, elementType, arrayBegin);

  EmitCXXAggrConstructorCall(ctor, numElements, arrayBegin, E, zeroInitialize);
}

/// EmitCXXAggrConstructorCall - Emit a loop to call a particular
/// constructor for each of several members of an array.
///
/// \param ctor the constructor to call for each element
/// \param numElements the number of elements in the array;
///   may be zero
/// \param arrayBegin a T*, where T is the type constructed by ctor
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *ctor,
                                                 llvm::Value *numElements,
                                                 llvm::Value *arrayBegin,
                                                 const CXXConstructExpr *E,
                                                 bool zeroInitialize) {

  // It's legal for numElements to be zero.  This can happen both
  // dynamically, because x can be zero in 'new A[x]', and statically,
  // because of GCC extensions that permit zero-length arrays.  There
  // are probably legitimate places where we could assume that this
  // doesn't happen, but it's not clear that it's worth it.
  llvm::BranchInst *zeroCheckBranch = nullptr;

  // Optimize for a constant count.
  llvm::ConstantInt *constantCount
    = dyn_cast<llvm::ConstantInt>(numElements);
  if (constantCount) {
    // Just skip out if the constant count is zero.
    if (constantCount->isZero()) return;

  // Otherwise, emit the check.
  } else {
    llvm::BasicBlock *loopBB = createBasicBlock("new.ctorloop");
    llvm::Value *iszero = Builder.CreateIsNull(numElements, "isempty");
    zeroCheckBranch = Builder.CreateCondBr(iszero, loopBB, loopBB);
    EmitBlock(loopBB);
  }
      
  // Find the end of the array.
  llvm::Value *arrayEnd = Builder.CreateInBoundsGEP(arrayBegin, numElements,
                                                    "arrayctor.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *entryBB = Builder.GetInsertBlock();
  llvm::BasicBlock *loopBB = createBasicBlock("arrayctor.loop");
  EmitBlock(loopBB);
  llvm::PHINode *cur = Builder.CreatePHI(arrayBegin->getType(), 2,
                                         "arrayctor.cur");
  cur->addIncoming(arrayBegin, entryBB);

  // Inside the loop body, emit the constructor call on the array element.

  QualType type = getContext().getTypeDeclType(ctor->getParent());

  // Zero initialize the storage, if requested.
  if (zeroInitialize)
    EmitNullInitialization(cur, type);
  
  // C++ [class.temporary]p4: 
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array. 
  // If the constructor has one or more default arguments, the destruction of 
  // every temporary created in a default argument expression is sequenced 
  // before the construction of the next array element, if any.
  
  {
    RunCleanupsScope Scope(*this);

    // Evaluate the constructor and its arguments in a regular
    // partial-destroy cleanup.
    if (getLangOpts().Exceptions &&
        !ctor->getParent()->hasTrivialDestructor()) {
      Destroyer *destroyer = destroyCXXObject;
      pushRegularPartialArrayCleanup(arrayBegin, cur, type, *destroyer);
    }

    EmitCXXConstructorCall(ctor, Ctor_Complete, /*ForVirtualBase=*/false,
                           /*Delegating=*/false, cur, E);
  }

  // Go to the next element.
  llvm::Value *next =
    Builder.CreateInBoundsGEP(cur, llvm::ConstantInt::get(SizeTy, 1),
                              "arrayctor.next");
  cur->addIncoming(next, Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *done = Builder.CreateICmpEQ(next, arrayEnd, "arrayctor.done");
  llvm::BasicBlock *contBB = createBasicBlock("arrayctor.cont");
  Builder.CreateCondBr(done, contBB, loopBB);

  // Patch the earlier check to skip over the loop.
  if (zeroCheckBranch) zeroCheckBranch->setSuccessor(0, contBB);

  EmitBlock(contBB);
}

void CodeGenFunction::destroyCXXObject(CodeGenFunction &CGF,
                                       llvm::Value *addr,
                                       QualType type) {
  const RecordType *rtype = type->castAs<RecordType>();
  const CXXRecordDecl *record = cast<CXXRecordDecl>(rtype->getDecl());
  const CXXDestructorDecl *dtor = record->getDestructor();
  assert(!dtor->isTrivial());
  CGF.EmitCXXDestructorCall(dtor, Dtor_Complete, /*for vbase*/ false,
                            /*Delegating=*/false, addr);
}

void CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D,
                                             CXXCtorType Type,
                                             bool ForVirtualBase,
                                             bool Delegating, llvm::Value *This,
                                             const CXXConstructExpr *E) {
  // If this is a trivial constructor, just emit what's needed.
  if (D->isTrivial() && !D->getParent()->mayInsertExtraPadding()) {
    if (E->getNumArgs() == 0) {
      // Trivial default constructor, no codegen required.
      assert(D->isDefaultConstructor() &&
             "trivial 0-arg ctor not a default ctor");
      return;
    }

    assert(E->getNumArgs() == 1 && "unexpected argcount for trivial ctor");
    assert(D->isCopyOrMoveConstructor() &&
           "trivial 1-arg ctor not a copy/move ctor");

    const Expr *Arg = E->getArg(0);
    QualType Ty = Arg->getType();
    llvm::Value *Src = EmitLValue(Arg).getAddress();
    EmitAggregateCopy(This, Src, Ty);
    return;
  }

  // C++11 [class.mfct.non-static]p2:
  //   If a non-static member function of a class X is called for an object that
  //   is not of type X, or of a type derived from X, the behavior is undefined.
  // FIXME: Provide a source location here.
  EmitTypeCheck(CodeGenFunction::TCK_ConstructorCall, SourceLocation(), This,
                getContext().getRecordType(D->getParent()));

  CallArgList Args;

  // Push the this ptr.
  Args.add(RValue::get(This), D->getThisType(getContext()));

  // Add the rest of the user-supplied arguments.
  const FunctionProtoType *FPT = D->getType()->castAs<FunctionProtoType>();
  EmitCallArgs(Args, FPT, E->arg_begin(), E->arg_end(), E->getConstructor());

  // Insert any ABI-specific implicit constructor arguments.
  unsigned ExtraArgs = CGM.getCXXABI().addImplicitConstructorArgs(
      *this, D, Type, ForVirtualBase, Delegating, Args);

  // Emit the call.
  llvm::Value *Callee = CGM.getAddrOfCXXStructor(D, getFromCtorType(Type));
  const CGFunctionInfo &Info =
      CGM.getTypes().arrangeCXXConstructorCall(Args, D, Type, ExtraArgs);
  EmitCall(Info, Callee, ReturnValueSlot(), Args, D);
}

void
CodeGenFunction::EmitSynthesizedCXXCopyCtorCall(const CXXConstructorDecl *D,
                                        llvm::Value *This, llvm::Value *Src,
                                        const CXXConstructExpr *E) {
  if (D->isTrivial() &&
      !D->getParent()->mayInsertExtraPadding()) {
    assert(E->getNumArgs() == 1 && "unexpected argcount for trivial ctor");
    assert(D->isCopyOrMoveConstructor() &&
           "trivial 1-arg ctor not a copy/move ctor");
    EmitAggregateCopy(This, Src, E->arg_begin()->getType());
    return;
  }
  llvm::Value *Callee = CGM.getAddrOfCXXStructor(D, StructorType::Complete);
  assert(D->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  
  const FunctionProtoType *FPT = D->getType()->castAs<FunctionProtoType>();
  
  CallArgList Args;
  
  // Push the this ptr.
  Args.add(RValue::get(This), D->getThisType(getContext()));
  
  // Push the src ptr.
  QualType QT = *(FPT->param_type_begin());
  llvm::Type *t = CGM.getTypes().ConvertType(QT);
  Src = Builder.CreateBitCast(Src, t);
  Args.add(RValue::get(Src), QT);

  // Skip over first argument (Src).
  EmitCallArgs(Args, FPT, E->arg_begin() + 1, E->arg_end(), E->getConstructor(),
               /*ParamsToSkip*/ 1);

  EmitCall(CGM.getTypes().arrangeCXXMethodCall(Args, FPT, RequiredArgs::All),
           Callee, ReturnValueSlot(), Args, D);
}

void
CodeGenFunction::EmitDelegateCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                                CXXCtorType CtorType,
                                                const FunctionArgList &Args,
                                                SourceLocation Loc) {
  CallArgList DelegateArgs;

  FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
  assert(I != E && "no parameters to constructor");

  // this
  DelegateArgs.add(RValue::get(LoadCXXThis()), (*I)->getType());
  ++I;

  // vtt
  if (llvm::Value *VTT = GetVTTParameter(GlobalDecl(Ctor, CtorType),
                                         /*ForVirtualBase=*/false,
                                         /*Delegating=*/true)) {
    QualType VoidPP = getContext().getPointerType(getContext().VoidPtrTy);
    DelegateArgs.add(RValue::get(VTT), VoidPP);

    if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
      assert(I != E && "cannot skip vtt parameter, already done with args");
      assert((*I)->getType() == VoidPP && "skipping parameter not of vtt type");
      ++I;
    }
  }

  // Explicit arguments.
  for (; I != E; ++I) {
    const VarDecl *param = *I;
    // FIXME: per-argument source location
    EmitDelegateCallArg(DelegateArgs, param, Loc);
  }

  llvm::Value *Callee =
      CGM.getAddrOfCXXStructor(Ctor, getFromCtorType(CtorType));
  EmitCall(CGM.getTypes()
               .arrangeCXXStructorDeclaration(Ctor, getFromCtorType(CtorType)),
           Callee, ReturnValueSlot(), DelegateArgs, Ctor);
}

namespace {
  struct CallDelegatingCtorDtor : EHScopeStack::Cleanup {
    const CXXDestructorDecl *Dtor;
    llvm::Value *Addr;
    CXXDtorType Type;

    CallDelegatingCtorDtor(const CXXDestructorDecl *D, llvm::Value *Addr,
                           CXXDtorType Type)
      : Dtor(D), Addr(Addr), Type(Type) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      CGF.EmitCXXDestructorCall(Dtor, Type, /*ForVirtualBase=*/false,
                                /*Delegating=*/true, Addr);
    }
  };
}

void
CodeGenFunction::EmitDelegatingCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                                  const FunctionArgList &Args) {
  assert(Ctor->isDelegatingConstructor());

  llvm::Value *ThisPtr = LoadCXXThis();

  QualType Ty = getContext().getTagDeclType(Ctor->getParent());
  CharUnits Alignment = getContext().getTypeAlignInChars(Ty);
  AggValueSlot AggSlot =
    AggValueSlot::forAddr(ThisPtr, Alignment, Qualifiers(),
                          AggValueSlot::IsDestructed,
                          AggValueSlot::DoesNotNeedGCBarriers,
                          AggValueSlot::IsNotAliased);

  EmitAggExpr(Ctor->init_begin()[0]->getInit(), AggSlot);

  const CXXRecordDecl *ClassDecl = Ctor->getParent();
  if (CGM.getLangOpts().Exceptions && !ClassDecl->hasTrivialDestructor()) {
    CXXDtorType Type =
      CurGD.getCtorType() == Ctor_Complete ? Dtor_Complete : Dtor_Base;

    EHStack.pushCleanup<CallDelegatingCtorDtor>(EHCleanup,
                                                ClassDecl->getDestructor(),
                                                ThisPtr, Type);
  }
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            bool ForVirtualBase,
                                            bool Delegating,
                                            llvm::Value *This) {
  CGM.getCXXABI().EmitDestructorCall(*this, DD, Type, ForVirtualBase,
                                     Delegating, This);
}

namespace {
  struct CallLocalDtor : EHScopeStack::Cleanup {
    const CXXDestructorDecl *Dtor;
    llvm::Value *Addr;

    CallLocalDtor(const CXXDestructorDecl *D, llvm::Value *Addr)
      : Dtor(D), Addr(Addr) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                /*ForVirtualBase=*/false,
                                /*Delegating=*/false, Addr);
    }
  };
}

void CodeGenFunction::PushDestructorCleanup(const CXXDestructorDecl *D,
                                            llvm::Value *Addr) {
  EHStack.pushCleanup<CallLocalDtor>(NormalAndEHCleanup, D, Addr);
}

void CodeGenFunction::PushDestructorCleanup(QualType T, llvm::Value *Addr) {
  CXXRecordDecl *ClassDecl = T->getAsCXXRecordDecl();
  if (!ClassDecl) return;
  if (ClassDecl->hasTrivialDestructor()) return;

  const CXXDestructorDecl *D = ClassDecl->getDestructor();
  assert(D && D->isUsed() && "destructor not marked as used!");
  PushDestructorCleanup(D, Addr);
}

void
CodeGenFunction::InitializeVTablePointer(BaseSubobject Base, 
                                         const CXXRecordDecl *NearestVBase,
                                         CharUnits OffsetFromNearestVBase,
                                         const CXXRecordDecl *VTableClass) {
  // Compute the address point.
  bool NeedsVirtualOffset;
  llvm::Value *VTableAddressPoint =
      CGM.getCXXABI().getVTableAddressPointInStructor(
          *this, VTableClass, Base, NearestVBase, NeedsVirtualOffset);
  if (!VTableAddressPoint)
    return;

  // Compute where to store the address point.
  llvm::Value *VirtualOffset = nullptr;
  CharUnits NonVirtualOffset = CharUnits::Zero();
  
  if (NeedsVirtualOffset) {
    // We need to use the virtual base offset offset because the virtual base
    // might have a different offset in the most derived class.
    VirtualOffset = CGM.getCXXABI().GetVirtualBaseClassOffset(*this,
                                                              LoadCXXThis(),
                                                              VTableClass,
                                                              NearestVBase);
    NonVirtualOffset = OffsetFromNearestVBase;
  } else {
    // We can just use the base offset in the complete class.
    NonVirtualOffset = Base.getBaseOffset();
  }
  
  // Apply the offsets.
  llvm::Value *VTableField = LoadCXXThis();
  
  if (!NonVirtualOffset.isZero() || VirtualOffset)
    VTableField = ApplyNonVirtualAndVirtualOffset(*this, VTableField, 
                                                  NonVirtualOffset,
                                                  VirtualOffset);

  // Finally, store the address point. Use the same LLVM types as the field to
  // support optimization.
  llvm::Type *VTablePtrTy =
      llvm::FunctionType::get(CGM.Int32Ty, /*isVarArg=*/true)
          ->getPointerTo()
          ->getPointerTo();
  VTableField = Builder.CreateBitCast(VTableField, VTablePtrTy->getPointerTo());
  VTableAddressPoint = Builder.CreateBitCast(VTableAddressPoint, VTablePtrTy);
  llvm::StoreInst *Store = Builder.CreateStore(VTableAddressPoint, VTableField);
  CGM.DecorateInstruction(Store, CGM.getTBAAInfoForVTablePtr());
}

void
CodeGenFunction::InitializeVTablePointers(BaseSubobject Base, 
                                          const CXXRecordDecl *NearestVBase,
                                          CharUnits OffsetFromNearestVBase,
                                          bool BaseIsNonVirtualPrimaryBase,
                                          const CXXRecordDecl *VTableClass,
                                          VisitedVirtualBasesSetTy& VBases) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    InitializeVTablePointer(Base, NearestVBase, OffsetFromNearestVBase,
                            VTableClass);
  }
  
  const CXXRecordDecl *RD = Base.getBase();

  // Traverse bases.
  for (const auto &I : RD->bases()) {
    CXXRecordDecl *BaseDecl
      = cast<CXXRecordDecl>(I.getType()->getAs<RecordType>()->getDecl());

    // Ignore classes without a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    CharUnits BaseOffset;
    CharUnits BaseOffsetFromNearestVBase;
    bool BaseDeclIsNonVirtualPrimaryBase;

    if (I.isVirtual()) {
      // Check if we've visited this virtual base before.
      if (!VBases.insert(BaseDecl).second)
        continue;

      const ASTRecordLayout &Layout = 
        getContext().getASTRecordLayout(VTableClass);

      BaseOffset = Layout.getVBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase = CharUnits::Zero();
      BaseDeclIsNonVirtualPrimaryBase = false;
    } else {
      const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

      BaseOffset = Base.getBaseOffset() + Layout.getBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase = 
        OffsetFromNearestVBase + Layout.getBaseClassOffset(BaseDecl);
      BaseDeclIsNonVirtualPrimaryBase = Layout.getPrimaryBase() == BaseDecl;
    }
    
    InitializeVTablePointers(BaseSubobject(BaseDecl, BaseOffset), 
                             I.isVirtual() ? BaseDecl : NearestVBase,
                             BaseOffsetFromNearestVBase,
                             BaseDeclIsNonVirtualPrimaryBase, 
                             VTableClass, VBases);
  }
}

void CodeGenFunction::InitializeVTablePointers(const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  VisitedVirtualBasesSetTy VBases;
  InitializeVTablePointers(BaseSubobject(RD, CharUnits::Zero()), 
                           /*NearestVBase=*/nullptr,
                           /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                           /*BaseIsNonVirtualPrimaryBase=*/false, RD, VBases);

  if (RD->getNumVBases())
    CGM.getCXXABI().initializeHiddenVirtualInheritanceMembers(*this, RD);
}

llvm::Value *CodeGenFunction::GetVTablePtr(llvm::Value *This,
                                           llvm::Type *Ty) {
  llvm::Value *VTablePtrSrc = Builder.CreateBitCast(This, Ty->getPointerTo());
  llvm::Instruction *VTable = Builder.CreateLoad(VTablePtrSrc, "vtable");
  CGM.DecorateInstruction(VTable, CGM.getTBAAInfoForVTablePtr());
  return VTable;
}


// FIXME: Ideally Expr::IgnoreParenNoopCasts should do this, but it doesn't do
// quite what we want.
static const Expr *skipNoOpCastsAndParens(const Expr *E) {
  while (true) {
    if (const ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
      E = PE->getSubExpr();
      continue;
    }

    if (const CastExpr *CE = dyn_cast<CastExpr>(E)) {
      if (CE->getCastKind() == CK_NoOp) {
        E = CE->getSubExpr();
        continue;
      }
    }
    if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
      if (UO->getOpcode() == UO_Extension) {
        E = UO->getSubExpr();
        continue;
      }
    }
    return E;
  }
}

bool
CodeGenFunction::CanDevirtualizeMemberFunctionCall(const Expr *Base,
                                                   const CXXMethodDecl *MD) {
  // When building with -fapple-kext, all calls must go through the vtable since
  // the kernel linker can do runtime patching of vtables.
  if (getLangOpts().AppleKext)
    return false;

  // If the most derived class is marked final, we know that no subclass can
  // override this member function and so we can devirtualize it. For example:
  //
  // struct A { virtual void f(); }
  // struct B final : A { };
  //
  // void f(B *b) {
  //   b->f();
  // }
  //
  const CXXRecordDecl *MostDerivedClassDecl = Base->getBestDynamicClassType();
  if (MostDerivedClassDecl->hasAttr<FinalAttr>())
    return true;

  // If the member function is marked 'final', we know that it can't be
  // overridden and can therefore devirtualize it.
  if (MD->hasAttr<FinalAttr>())
    return true;

  // Similarly, if the class itself is marked 'final' it can't be overridden
  // and we can therefore devirtualize the member function call.
  if (MD->getParent()->hasAttr<FinalAttr>())
    return true;

  Base = skipNoOpCastsAndParens(Base);
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Base)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      // This is a record decl. We know the type and can devirtualize it.
      return VD->getType()->isRecordType();
    }
    
    return false;
  }

  // We can devirtualize calls on an object accessed by a class member access
  // expression, since by C++11 [basic.life]p6 we know that it can't refer to
  // a derived class object constructed in the same location.
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(Base))
    if (const ValueDecl *VD = dyn_cast<ValueDecl>(ME->getMemberDecl()))
      return VD->getType()->isRecordType();

  // We can always devirtualize calls on temporary object expressions.
  if (isa<CXXConstructExpr>(Base))
    return true;
  
  // And calls on bound temporaries.
  if (isa<CXXBindTemporaryExpr>(Base))
    return true;
  
  // Check if this is a call expr that returns a record type.
  if (const CallExpr *CE = dyn_cast<CallExpr>(Base))
    return CE->getCallReturnType()->isRecordType();

  // We can't devirtualize the call.
  return false;
}

void CodeGenFunction::EmitForwardingCallToLambda(
                                      const CXXMethodDecl *callOperator,
                                      CallArgList &callArgs) {
  // Get the address of the call operator.
  const CGFunctionInfo &calleeFnInfo =
    CGM.getTypes().arrangeCXXMethodDeclaration(callOperator);
  llvm::Value *callee =
    CGM.GetAddrOfFunction(GlobalDecl(callOperator),
                          CGM.getTypes().GetFunctionType(calleeFnInfo));

  // Prepare the return slot.
  const FunctionProtoType *FPT =
    callOperator->getType()->castAs<FunctionProtoType>();
  QualType resultType = FPT->getReturnType();
  ReturnValueSlot returnSlot;
  if (!resultType->isVoidType() &&
      calleeFnInfo.getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      !hasScalarEvaluationKind(calleeFnInfo.getReturnType()))
    returnSlot = ReturnValueSlot(ReturnValue, resultType.isVolatileQualified());

  // We don't need to separately arrange the call arguments because
  // the call can't be variadic anyway --- it's impossible to forward
  // variadic arguments.
  
  // Now emit our call.
  RValue RV = EmitCall(calleeFnInfo, callee, returnSlot,
                       callArgs, callOperator);

  // If necessary, copy the returned value into the slot.
  if (!resultType->isVoidType() && returnSlot.isNull())
    EmitReturnOfRValue(RV, resultType);
  else
    EmitBranchThroughCleanup(ReturnBlock);
}

void CodeGenFunction::EmitLambdaBlockInvokeBody() {
  const BlockDecl *BD = BlockInfo->getBlockDecl();
  const VarDecl *variable = BD->capture_begin()->getVariable();
  const CXXRecordDecl *Lambda = variable->getType()->getAsCXXRecordDecl();

  // Start building arguments for forwarding call
  CallArgList CallArgs;

  QualType ThisType = getContext().getPointerType(getContext().getRecordType(Lambda));
  llvm::Value *ThisPtr = GetAddrOfBlockDecl(variable, false);
  CallArgs.add(RValue::get(ThisPtr), ThisType);

  // Add the rest of the parameters.
  for (auto param : BD->params())
    EmitDelegateCallArg(CallArgs, param, param->getLocStart());

  assert(!Lambda->isGenericLambda() && 
            "generic lambda interconversion to block not implemented");
  EmitForwardingCallToLambda(Lambda->getLambdaCallOperator(), CallArgs);
}

void CodeGenFunction::EmitLambdaToBlockPointerBody(FunctionArgList &Args) {
  if (cast<CXXMethodDecl>(CurCodeDecl)->isVariadic()) {
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator forward.
    CGM.ErrorUnsupported(CurCodeDecl, "lambda conversion to variadic function");
    return;
  }

  EmitFunctionBody(Args, cast<FunctionDecl>(CurGD.getDecl())->getBody());
}

void CodeGenFunction::EmitLambdaDelegatingInvokeBody(const CXXMethodDecl *MD) {
  const CXXRecordDecl *Lambda = MD->getParent();

  // Start building arguments for forwarding call
  CallArgList CallArgs;

  QualType ThisType = getContext().getPointerType(getContext().getRecordType(Lambda));
  llvm::Value *ThisPtr = llvm::UndefValue::get(getTypes().ConvertType(ThisType));
  CallArgs.add(RValue::get(ThisPtr), ThisType);

  // Add the rest of the parameters.
  for (auto Param : MD->params())
    EmitDelegateCallArg(CallArgs, Param, Param->getLocStart());

  const CXXMethodDecl *CallOp = Lambda->getLambdaCallOperator();
  // For a generic lambda, find the corresponding call operator specialization
  // to which the call to the static-invoker shall be forwarded.
  if (Lambda->isGenericLambda()) {
    assert(MD->isFunctionTemplateSpecialization());
    const TemplateArgumentList *TAL = MD->getTemplateSpecializationArgs();
    FunctionTemplateDecl *CallOpTemplate = CallOp->getDescribedFunctionTemplate();
    void *InsertPos = nullptr;
    FunctionDecl *CorrespondingCallOpSpecialization = 
        CallOpTemplate->findSpecialization(TAL->asArray(), InsertPos);
    assert(CorrespondingCallOpSpecialization);
    CallOp = cast<CXXMethodDecl>(CorrespondingCallOpSpecialization);
  }
  EmitForwardingCallToLambda(CallOp, CallArgs);
}

void CodeGenFunction::EmitLambdaStaticInvokeFunction(const CXXMethodDecl *MD) {
  if (MD->isVariadic()) {
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator forward.
    CGM.ErrorUnsupported(MD, "lambda conversion to variadic function");
    return;
  }

  EmitLambdaDelegatingInvokeBody(MD);
}
