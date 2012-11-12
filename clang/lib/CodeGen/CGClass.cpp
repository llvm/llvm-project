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
#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtCXX.h"
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
    return 0;
  
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
  assert(!nonVirtualOffset.isZero() || virtualOffset != 0);

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

llvm::Value *
CodeGenFunction::GetAddressOfBaseClass(llvm::Value *Value, 
                                       const CXXRecordDecl *Derived,
                                       CastExpr::path_const_iterator PathBegin,
                                       CastExpr::path_const_iterator PathEnd,
                                       bool NullCheckValue) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  CastExpr::path_const_iterator Start = PathBegin;
  const CXXRecordDecl *VBase = 0;
  
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
    VBase = 0; // we no longer have a virtual step
  }

  // Get the base pointer type.
  llvm::Type *BasePtrTy = 
    ConvertType((PathEnd[-1])->getType())->getPointerTo();

  // If the static offset is zero and we don't have a virtual step,
  // just do a bitcast; null checks are unnecessary.
  if (NonVirtualOffset.isZero() && !VBase) {
    return Builder.CreateBitCast(Value, BasePtrTy);
  }    

  llvm::BasicBlock *origBB = 0;
  llvm::BasicBlock *endBB = 0;
  
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

  // Compute the virtual offset.
  llvm::Value *VirtualOffset = 0;
  if (VBase) {
    VirtualOffset = GetVirtualBaseClassOffset(Value, Derived, VBase);
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
  
  llvm::BasicBlock *CastNull = 0;
  llvm::BasicBlock *CastNotNull = 0;
  llvm::BasicBlock *CastEnd = 0;
  
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
                             
/// GetVTTParameter - Return the VTT parameter that should be passed to a
/// base constructor/destructor with virtual bases.
static llvm::Value *GetVTTParameter(CodeGenFunction &CGF, GlobalDecl GD,
                                    bool ForVirtualBase) {
  if (!CodeGenVTables::needsVTTParameter(GD)) {
    // This constructor/destructor does not need a VTT parameter.
    return 0;
  }
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(CGF.CurFuncDecl)->getParent();
  const CXXRecordDecl *Base = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  llvm::Value *VTT;

  uint64_t SubVTTIndex;

  // If the record matches the base, this is the complete ctor/dtor
  // variant calling the base variant in a class with virtual bases.
  if (RD == Base) {
    assert(!CodeGenVTables::needsVTTParameter(CGF.CurGD) &&
           "doing no-op VTT offset in base dtor/ctor?");
    assert(!ForVirtualBase && "Can't have same class as virtual base!");
    SubVTTIndex = 0;
  } else {
    const ASTRecordLayout &Layout = 
      CGF.getContext().getASTRecordLayout(RD);
    CharUnits BaseOffset = ForVirtualBase ? 
      Layout.getVBaseClassOffset(Base) : 
      Layout.getBaseClassOffset(Base);

    SubVTTIndex = 
      CGF.CGM.getVTables().getSubVTTIndex(RD, BaseSubobject(Base, BaseOffset));
    assert(SubVTTIndex != 0 && "Sub-VTT index must be greater than zero!");
  }
  
  if (CodeGenVTables::needsVTTParameter(CGF.CurGD)) {
    // A VTT parameter was passed to the constructor, use it.
    VTT = CGF.LoadCXXVTT();
    VTT = CGF.Builder.CreateConstInBoundsGEP1_64(VTT, SubVTTIndex);
  } else {
    // We're the complete constructor, so get the VTT by name.
    VTT = CGF.CGM.getVTables().GetAddrOfVTT(RD);
    VTT = CGF.Builder.CreateConstInBoundsGEP2_64(VTT, 0, SubVTTIndex);
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

    void Emit(CodeGenFunction &CGF, Flags flags) {
      const CXXRecordDecl *DerivedClass =
        cast<CXXMethodDecl>(CGF.CurCodeDecl)->getParent();

      const CXXDestructorDecl *D = BaseClass->getDestructor();
      llvm::Value *Addr = 
        CGF.GetAddressOfDirectBaseInCompleteClass(CGF.LoadCXXThis(),
                                                  DerivedClass, BaseClass,
                                                  BaseIsVirtual);
      CGF.EmitCXXDestructorCall(D, Dtor_Base, BaseIsVirtual, Addr);
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
    { // Scope for Cleanups.
      CodeGenFunction::RunCleanupsScope Cleanups(CGF);

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

      if (!CGF.hasAggregateLLVMType(T)) {
        CGF.EmitScalarInit(Init, /*decl*/ 0, LV, false);
      } else if (T->isAnyComplexType()) {
        CGF.EmitComplexExprIntoAddr(Init, LV.getAddress(),
                                    LV.isVolatileQualified());
      } else {
        AggValueSlot Slot =
          AggValueSlot::forLValue(LV,
                                  AggValueSlot::IsDestructed,
                                  AggValueSlot::DoesNotNeedGCBarriers,
                                  AggValueSlot::IsNotAliased);

        CGF.EmitAggExpr(Init, Slot);
      }
    }

    // Now, outside of the initializer cleanup scope, destroy the backing array
    // for a std::initializer_list member.
    CGF.MaybeEmitStdInitializerListCleanup(LV.getAddress(), Init);

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
  
  {
    CodeGenFunction::RunCleanupsScope Cleanups(CGF);
    
    // Inside the loop body recurse to emit the inner loop or, eventually, the
    // constructor call.
    EmitAggMemberInitializer(CGF, LHS, Init, ArrayIndexVar,
                             Array->getElementType(), ArrayIndexes, Index + 1);
  }
  
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

namespace {
  struct CallMemberDtor : EHScopeStack::Cleanup {
    llvm::Value *V;
    CXXDestructorDecl *Dtor;

    CallMemberDtor(llvm::Value *V, CXXDestructorDecl *Dtor)
      : V(V), Dtor(Dtor) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                                V);
    }
  };
}

static void EmitMemberInitializer(CodeGenFunction &CGF,
                                  const CXXRecordDecl *ClassDecl,
                                  CXXCtorInitializer *MemberInit,
                                  const CXXConstructorDecl *Constructor,
                                  FunctionArgList &Args) {
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
    IndirectFieldDecl::chain_iterator I = IndirectField->chain_begin(),
      IEnd = IndirectField->chain_end();
    for ( ; I != IEnd; ++I)
      LHS = CGF.EmitLValueForFieldInitialization(LHS, cast<FieldDecl>(*I));
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
  if (Array && Constructor->isImplicitlyDefined() &&
      Constructor->isCopyOrMoveConstructor()) {
    QualType BaseElementTy = CGF.getContext().getBaseElementType(Array);
    CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(MemberInit->getInit());
    if (BaseElementTy.isPODType(CGF.getContext()) ||
        (CE && CE->getConstructor()->isTrivial())) {
      // Find the source pointer. We know it's the last argument because
      // we know we're in an implicit copy constructor.
      unsigned SrcArgIndex = Args.size() - 1;
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

void CodeGenFunction::EmitInitializerForField(FieldDecl *Field,
                                              LValue LHS, Expr *Init,
                                             ArrayRef<VarDecl *> ArrayIndexes) {
  QualType FieldType = Field->getType();
  if (!hasAggregateLLVMType(FieldType)) {
    if (LHS.isSimple()) {
      EmitExprAsInit(Init, Field, LHS, false);
    } else {
      RValue RHS = RValue::get(EmitScalarExpr(Init));
      EmitStoreThroughLValue(RHS, LHS);
    }
  } else if (FieldType->isAnyComplexType()) {
    EmitComplexExprIntoAddr(Init, LHS.getAddress(), LHS.isVolatileQualified());
  } else {
    llvm::Value *ArrayIndexVar = 0;
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
    
    if (!CGM.getLangOpts().Exceptions)
      return;

    // FIXME: If we have an array of classes w/ non-trivial destructors, 
    // we need to destroy in reverse order of construction along the exception
    // path.
    const RecordType *RT = FieldType->getAs<RecordType>();
    if (!RT)
      return;
    
    CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    if (!RD->hasTrivialDestructor())
      EHStack.pushCleanup<CallMemberDtor>(EHCleanup, LHS.getAddress(),
                                          RD->getDestructor());
  }
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

/// EmitConstructorBody - Emits the body of the current constructor.
void CodeGenFunction::EmitConstructorBody(FunctionArgList &Args) {
  const CXXConstructorDecl *Ctor = cast<CXXConstructorDecl>(CurGD.getDecl());
  CXXCtorType CtorType = CurGD.getCtorType();

  // Before we go any further, try the complete->base constructor
  // delegation optimization.
  if (CtorType == Ctor_Complete && IsConstructorDelegationValid(Ctor) &&
      CGM.getContext().getTargetInfo().getCXXABI() != CXXABI_Microsoft) {
    if (CGDebugInfo *DI = getDebugInfo()) 
      DI->EmitLocation(Builder, Ctor->getLocEnd());
    EmitDelegateCXXConstructorCall(Ctor, Ctor_Base, Args);
    return;
  }

  Stmt *Body = Ctor->getBody();

  // Enter the function-try-block before the constructor prologue if
  // applicable.
  bool IsTryBody = (Body && isa<CXXTryStmt>(Body));
  if (IsTryBody)
    EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);

  EHScopeStack::stable_iterator CleanupDepth = EHStack.stable_begin();

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
  PopCleanupBlocks(CleanupDepth);

  if (IsTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), true);
}

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD,
                                       CXXCtorType CtorType,
                                       FunctionArgList &Args) {
  if (CD->isDelegatingConstructor())
    return EmitDelegatingCXXConstructorCall(CD, Args);

  const CXXRecordDecl *ClassDecl = CD->getParent();

  SmallVector<CXXCtorInitializer *, 8> MemberInitializers;
  
  for (CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
       E = CD->init_end();
       B != E; ++B) {
    CXXCtorInitializer *Member = (*B);
    
    if (Member->isBaseInitializer()) {
      EmitBaseInitializer(*this, ClassDecl, Member, CtorType);
    } else {
      assert(Member->isAnyMemberInitializer() &&
            "Delegating initializer on non-delegating constructor");
      MemberInitializers.push_back(Member);
    }
  }

  InitializeVTablePointers(ClassDecl);

  for (unsigned I = 0, E = MemberInitializers.size(); I != E; ++I)
    EmitMemberInitializer(*this, ClassDecl, MemberInitializers[I], CD, Args);
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
  for (CXXRecordDecl::field_iterator I = BaseClassDecl->field_begin(),
       E = BaseClassDecl->field_end(); I != E; ++I) {
    const FieldDecl *Field = *I;
    
    if (!FieldHasTrivialDestructorBody(Context, Field))
      return false;
  }

  // Check non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = 
       BaseClassDecl->bases_begin(), E = BaseClassDecl->bases_end();
       I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *NonVirtualBase =
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());
    if (!HasTrivialDestructorBody(Context, NonVirtualBase,
                                  MostDerivedClassDecl))
      return false;
  }

  if (BaseClassDecl == MostDerivedClassDecl) {
    // Check virtual bases.
    for (CXXRecordDecl::base_class_const_iterator I = 
         BaseClassDecl->vbases_begin(), E = BaseClassDecl->vbases_end();
         I != E; ++I) {
      const CXXRecordDecl *VirtualBase =
        cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());
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
  for (CXXRecordDecl::field_iterator I = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); I != E; ++I) {
    const FieldDecl *Field = *I;

    if (!FieldHasTrivialDestructorBody(Context, Field))
      return false;
  }

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
                          LoadCXXThis());
    PopCleanupBlock();
    return;
  }

  Stmt *Body = Dtor->getBody();

  // If the body is a function-try-block, enter the try before
  // anything else.
  bool isTryBody = (Body && isa<CXXTryStmt>(Body));
  if (isTryBody)
    EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);

  // Enter the epilogue cleanups.
  RunCleanupsScope DtorEpilogue(*this);
  
  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.
  switch (DtorType) {
  case Dtor_Deleting: llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    // Enter the cleanup scopes for virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Complete);

    if (!isTryBody && CGM.getContext().getTargetInfo().getCXXABI() != CXXABI_Microsoft) {
      EmitCXXDestructorCall(Dtor, Dtor_Base, /*ForVirtualBase=*/false,
                            LoadCXXThis());
      break;
    }
    // Fallthrough: act like we're in the base variant.
      
  case Dtor_Base:
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
      CurFn->addFnAttr(llvm::Attributes::AlwaysInline);
    break;
  }

  // Jump out through the epilogue cleanups.
  DtorEpilogue.ForceCleanup();

  // Exit the try if applicable.
  if (isTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), true);
}

namespace {
  /// Call the operator delete associated with the current destructor.
  struct CallDtorDelete : EHScopeStack::Cleanup {
    CallDtorDelete() {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CGF.CurCodeDecl);
      const CXXRecordDecl *ClassDecl = Dtor->getParent();
      CGF.EmitDeleteCall(Dtor->getOperatorDelete(), CGF.LoadCXXThis(),
                         CGF.getContext().getTagDeclType(ClassDecl));
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

    void Emit(CodeGenFunction &CGF, Flags flags) {
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

/// EmitDtorEpilogue - Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes
/// in reverse order of their construction.
void CodeGenFunction::EnterDtorCleanups(const CXXDestructorDecl *DD,
                                        CXXDtorType DtorType) {
  assert(!DD->isTrivial() &&
         "Should not emit dtor epilogue for trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() && 
           "operator delete missing - EmitDtorEpilogue");
    EHStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
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
    for (CXXRecordDecl::base_class_const_iterator I = 
           ClassDecl->vbases_begin(), E = ClassDecl->vbases_end();
              I != E; ++I) {
      const CXXBaseSpecifier &Base = *I;
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
  for (CXXRecordDecl::base_class_const_iterator I = 
        ClassDecl->bases_begin(), E = ClassDecl->bases_end(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    
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
  SmallVector<const FieldDecl *, 16> FieldDecls;
  for (CXXRecordDecl::field_iterator I = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); I != E; ++I) {
    const FieldDecl *field = *I;
    QualType type = field->getType();
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind) continue;

    // Anonymous union members do not have their destructors called.
    const RecordType *RT = type->getAsUnionType();
    if (RT && RT->getDecl()->isAnonymousStructOrUnion()) continue;

    CleanupKind cleanupKind = getCleanupKind(dtorKind);
    EHStack.pushCleanup<DestroyField>(cleanupKind, field,
                                      getDestroyer(dtorKind),
                                      cleanupKind & EHCleanup);
  }
}

/// EmitCXXAggrConstructorCall - Emit a loop to call a particular
/// constructor for each of several members of an array.
///
/// \param ctor the constructor to call for each element
/// \param argBegin,argEnd the arguments to evaluate and pass to the
///   constructor
/// \param arrayType the type of the array to initialize
/// \param arrayBegin an arrayType*
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *ctor,
                                            const ConstantArrayType *arrayType,
                                            llvm::Value *arrayBegin,
                                          CallExpr::const_arg_iterator argBegin,
                                            CallExpr::const_arg_iterator argEnd,
                                            bool zeroInitialize) {
  QualType elementType;
  llvm::Value *numElements =
    emitArrayLength(arrayType, elementType, arrayBegin);

  EmitCXXAggrConstructorCall(ctor, numElements, arrayBegin,
                             argBegin, argEnd, zeroInitialize);
}

/// EmitCXXAggrConstructorCall - Emit a loop to call a particular
/// constructor for each of several members of an array.
///
/// \param ctor the constructor to call for each element
/// \param numElements the number of elements in the array;
///   may be zero
/// \param argBegin,argEnd the arguments to evaluate and pass to the
///   constructor
/// \param arrayBegin a T*, where T is the type constructed by ctor
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *ctor,
                                            llvm::Value *numElements,
                                            llvm::Value *arrayBegin,
                                         CallExpr::const_arg_iterator argBegin,
                                           CallExpr::const_arg_iterator argEnd,
                                            bool zeroInitialize) {

  // It's legal for numElements to be zero.  This can happen both
  // dynamically, because x can be zero in 'new A[x]', and statically,
  // because of GCC extensions that permit zero-length arrays.  There
  // are probably legitimate places where we could assume that this
  // doesn't happen, but it's not clear that it's worth it.
  llvm::BranchInst *zeroCheckBranch = 0;

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

    EmitCXXConstructorCall(ctor, Ctor_Complete, /*ForVirtualBase=*/ false,
                           cur, argBegin, argEnd);
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
                            addr);
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D,
                                        CXXCtorType Type, bool ForVirtualBase,
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {

  CGDebugInfo *DI = getDebugInfo();
  if (DI &&
      CGM.getCodeGenOpts().getDebugInfo() == CodeGenOptions::LimitedDebugInfo) {
    // If debug info for this class has not been emitted then this is the
    // right time to do so.
    const CXXRecordDecl *Parent = D->getParent();
    DI->getOrCreateRecordType(CGM.getContext().getTypeDeclType(Parent),
                              Parent->getLocation());
  }

  if (D->isTrivial()) {
    if (ArgBeg == ArgEnd) {
      // Trivial default constructor, no codegen required.
      assert(D->isDefaultConstructor() &&
             "trivial 0-arg ctor not a default ctor");
      return;
    }

    assert(ArgBeg + 1 == ArgEnd && "unexpected argcount for trivial ctor");
    assert(D->isCopyOrMoveConstructor() &&
           "trivial 1-arg ctor not a copy/move ctor");

    const Expr *E = (*ArgBeg);
    QualType Ty = E->getType();
    llvm::Value *Src = EmitLValue(E).getAddress();
    EmitAggregateCopy(This, Src, Ty);
    return;
  }

  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(D, Type), ForVirtualBase);
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  // FIXME: Provide a source location here.
  EmitCXXMemberCall(D, SourceLocation(), Callee, ReturnValueSlot(), This,
                    VTT, ArgBeg, ArgEnd);
}

void
CodeGenFunction::EmitSynthesizedCXXCopyCtorCall(const CXXConstructorDecl *D,
                                        llvm::Value *This, llvm::Value *Src,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  if (D->isTrivial()) {
    assert(ArgBeg + 1 == ArgEnd && "unexpected argcount for trivial ctor");
    assert(D->isCopyOrMoveConstructor() &&
           "trivial 1-arg ctor not a copy/move ctor");
    EmitAggregateCopy(This, Src, (*ArgBeg)->getType());
    return;
  }
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, 
                                                    clang::Ctor_Complete);
  assert(D->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  
  const FunctionProtoType *FPT = D->getType()->getAs<FunctionProtoType>();
  
  CallArgList Args;
  
  // Push the this ptr.
  Args.add(RValue::get(This), D->getThisType(getContext()));
  
  
  // Push the src ptr.
  QualType QT = *(FPT->arg_type_begin());
  llvm::Type *t = CGM.getTypes().ConvertType(QT);
  Src = Builder.CreateBitCast(Src, t);
  Args.add(RValue::get(Src), QT);
  
  // Skip over first argument (Src).
  ++ArgBeg;
  CallExpr::const_arg_iterator Arg = ArgBeg;
  for (FunctionProtoType::arg_type_iterator I = FPT->arg_type_begin()+1,
       E = FPT->arg_type_end(); I != E; ++I, ++Arg) {
    assert(Arg != ArgEnd && "Running over edge of argument list!");
    EmitCallArg(Args, *Arg, *I);
  }
  // Either we've emitted all the call args, or we have a call to a
  // variadic function.
  assert((Arg == ArgEnd || FPT->isVariadic()) &&
         "Extra arguments in non-variadic function!");
  // If we still have any arguments, emit them using the type of the argument.
  for (; Arg != ArgEnd; ++Arg) {
    QualType ArgType = Arg->getType();
    EmitCallArg(Args, *Arg, ArgType);
  }
  
  EmitCall(CGM.getTypes().arrangeCXXMethodCall(Args, FPT, RequiredArgs::All),
           Callee, ReturnValueSlot(), Args, D);
}

void
CodeGenFunction::EmitDelegateCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                                CXXCtorType CtorType,
                                                const FunctionArgList &Args) {
  CallArgList DelegateArgs;

  FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
  assert(I != E && "no parameters to constructor");

  // this
  DelegateArgs.add(RValue::get(LoadCXXThis()), (*I)->getType());
  ++I;

  // vtt
  if (llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(Ctor, CtorType),
                                         /*ForVirtualBase=*/false)) {
    QualType VoidPP = getContext().getPointerType(getContext().VoidPtrTy);
    DelegateArgs.add(RValue::get(VTT), VoidPP);

    if (CodeGenVTables::needsVTTParameter(CurGD)) {
      assert(I != E && "cannot skip vtt parameter, already done with args");
      assert((*I)->getType() == VoidPP && "skipping parameter not of vtt type");
      ++I;
    }
  }

  // Explicit arguments.
  for (; I != E; ++I) {
    const VarDecl *param = *I;
    EmitDelegateCallArg(DelegateArgs, param);
  }

  EmitCall(CGM.getTypes().arrangeCXXConstructorDeclaration(Ctor, CtorType),
           CGM.GetAddrOfCXXConstructor(Ctor, CtorType), 
           ReturnValueSlot(), DelegateArgs, Ctor);
}

namespace {
  struct CallDelegatingCtorDtor : EHScopeStack::Cleanup {
    const CXXDestructorDecl *Dtor;
    llvm::Value *Addr;
    CXXDtorType Type;

    CallDelegatingCtorDtor(const CXXDestructorDecl *D, llvm::Value *Addr,
                           CXXDtorType Type)
      : Dtor(D), Addr(Addr), Type(Type) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.EmitCXXDestructorCall(Dtor, Type, /*ForVirtualBase=*/false,
                                Addr);
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
                                            llvm::Value *This) {
  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(DD, Type), 
                                     ForVirtualBase);
  llvm::Value *Callee = 0;
  if (getLangOpts().AppleKext)
    Callee = BuildAppleKextVirtualDestructorCall(DD, Type, 
                                                 DD->getParent());
    
  if (!Callee)
    Callee = CGM.GetAddrOfCXXDestructor(DD, Type);
  
  // FIXME: Provide a source location here.
  EmitCXXMemberCall(DD, SourceLocation(), Callee, ReturnValueSlot(), This,
                    VTT, 0, 0);
}

namespace {
  struct CallLocalDtor : EHScopeStack::Cleanup {
    const CXXDestructorDecl *Dtor;
    llvm::Value *Addr;

    CallLocalDtor(const CXXDestructorDecl *D, llvm::Value *Addr)
      : Dtor(D), Addr(Addr) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                /*ForVirtualBase=*/false, Addr);
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

llvm::Value *
CodeGenFunction::GetVirtualBaseClassOffset(llvm::Value *This,
                                           const CXXRecordDecl *ClassDecl,
                                           const CXXRecordDecl *BaseClassDecl) {
  llvm::Value *VTablePtr = GetVTablePtr(This, Int8PtrTy);
  CharUnits VBaseOffsetOffset = 
    CGM.getVTableContext().getVirtualBaseOffsetOffset(ClassDecl, BaseClassDecl);
  
  llvm::Value *VBaseOffsetPtr = 
    Builder.CreateConstGEP1_64(VTablePtr, VBaseOffsetOffset.getQuantity(), 
                               "vbase.offset.ptr");
  llvm::Type *PtrDiffTy = 
    ConvertType(getContext().getPointerDiffType());
  
  VBaseOffsetPtr = Builder.CreateBitCast(VBaseOffsetPtr, 
                                         PtrDiffTy->getPointerTo());
                                         
  llvm::Value *VBaseOffset = Builder.CreateLoad(VBaseOffsetPtr, "vbase.offset");
  
  return VBaseOffset;
}

void
CodeGenFunction::InitializeVTablePointer(BaseSubobject Base, 
                                         const CXXRecordDecl *NearestVBase,
                                         CharUnits OffsetFromNearestVBase,
                                         llvm::Constant *VTable,
                                         const CXXRecordDecl *VTableClass) {
  const CXXRecordDecl *RD = Base.getBase();

  // Compute the address point.
  llvm::Value *VTableAddressPoint;

  // Check if we need to use a vtable from the VTT.
  if (CodeGenVTables::needsVTTParameter(CurGD) &&
      (RD->getNumVBases() || NearestVBase)) {
    // Get the secondary vpointer index.
    uint64_t VirtualPointerIndex = 
     CGM.getVTables().getSecondaryVirtualPointerIndex(VTableClass, Base);
    
    /// Load the VTT.
    llvm::Value *VTT = LoadCXXVTT();
    if (VirtualPointerIndex)
      VTT = Builder.CreateConstInBoundsGEP1_64(VTT, VirtualPointerIndex);

    // And load the address point from the VTT.
    VTableAddressPoint = Builder.CreateLoad(VTT);
  } else {
    uint64_t AddressPoint =
      CGM.getVTableContext().getVTableLayout(VTableClass).getAddressPoint(Base);
    VTableAddressPoint =
      Builder.CreateConstInBoundsGEP2_64(VTable, 0, AddressPoint);
  }

  // Compute where to store the address point.
  llvm::Value *VirtualOffset = 0;
  CharUnits NonVirtualOffset = CharUnits::Zero();
  
  if (CodeGenVTables::needsVTTParameter(CurGD) && NearestVBase) {
    // We need to use the virtual base offset offset because the virtual base
    // might have a different offset in the most derived class.
    VirtualOffset = GetVirtualBaseClassOffset(LoadCXXThis(), VTableClass, 
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

  // Finally, store the address point.
  llvm::Type *AddressPointPtrTy =
    VTableAddressPoint->getType()->getPointerTo();
  VTableField = Builder.CreateBitCast(VTableField, AddressPointPtrTy);
  llvm::StoreInst *Store = Builder.CreateStore(VTableAddressPoint, VTableField);
  CGM.DecorateInstruction(Store, CGM.getTBAAInfoForVTablePtr());
}

void
CodeGenFunction::InitializeVTablePointers(BaseSubobject Base, 
                                          const CXXRecordDecl *NearestVBase,
                                          CharUnits OffsetFromNearestVBase,
                                          bool BaseIsNonVirtualPrimaryBase,
                                          llvm::Constant *VTable,
                                          const CXXRecordDecl *VTableClass,
                                          VisitedVirtualBasesSetTy& VBases) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    InitializeVTablePointer(Base, NearestVBase, OffsetFromNearestVBase,
                            VTable, VTableClass);
  }
  
  const CXXRecordDecl *RD = Base.getBase();

  // Traverse bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(), 
       E = RD->bases_end(); I != E; ++I) {
    CXXRecordDecl *BaseDecl
      = cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Ignore classes without a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    CharUnits BaseOffset;
    CharUnits BaseOffsetFromNearestVBase;
    bool BaseDeclIsNonVirtualPrimaryBase;

    if (I->isVirtual()) {
      // Check if we've visited this virtual base before.
      if (!VBases.insert(BaseDecl))
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
                             I->isVirtual() ? BaseDecl : NearestVBase,
                             BaseOffsetFromNearestVBase,
                             BaseDeclIsNonVirtualPrimaryBase, 
                             VTable, VTableClass, VBases);
  }
}

void CodeGenFunction::InitializeVTablePointers(const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Get the VTable.
  llvm::Constant *VTable = CGM.getVTables().GetAddrOfVTable(RD);

  // Initialize the vtable pointers for this class and all of its bases.
  VisitedVirtualBasesSetTy VBases;
  InitializeVTablePointers(BaseSubobject(RD, CharUnits::Zero()), 
                           /*NearestVBase=*/0, 
                           /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                           /*BaseIsNonVirtualPrimaryBase=*/false, 
                           VTable, RD, VBases);
}

llvm::Value *CodeGenFunction::GetVTablePtr(llvm::Value *This,
                                           llvm::Type *Ty) {
  llvm::Value *VTablePtrSrc = Builder.CreateBitCast(This, Ty->getPointerTo());
  llvm::Instruction *VTable = Builder.CreateLoad(VTablePtrSrc, "vtable");
  CGM.DecorateInstruction(VTable, CGM.getTBAAInfoForVTablePtr());
  return VTable;
}

static const CXXRecordDecl *getMostDerivedClassDecl(const Expr *Base) {
  const Expr *E = Base;
  
  while (true) {
    E = E->IgnoreParens();
    if (const CastExpr *CE = dyn_cast<CastExpr>(E)) {
      if (CE->getCastKind() == CK_DerivedToBase || 
          CE->getCastKind() == CK_UncheckedDerivedToBase ||
          CE->getCastKind() == CK_NoOp) {
        E = CE->getSubExpr();
        continue;
      }
    }

    break;
  }

  QualType DerivedType = E->getType();
  if (const PointerType *PTy = DerivedType->getAs<PointerType>())
    DerivedType = PTy->getPointeeType();

  return cast<CXXRecordDecl>(DerivedType->castAs<RecordType>()->getDecl());
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

/// canDevirtualizeMemberFunctionCall - Checks whether the given virtual member
/// function call on the given expr can be devirtualized.
static bool canDevirtualizeMemberFunctionCall(const Expr *Base, 
                                              const CXXMethodDecl *MD) {
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
  const CXXRecordDecl *MostDerivedClassDecl = getMostDerivedClassDecl(Base);
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

static bool UseVirtualCall(ASTContext &Context,
                           const CXXOperatorCallExpr *CE,
                           const CXXMethodDecl *MD) {
  if (!MD->isVirtual())
    return false;
  
  // When building with -fapple-kext, all calls must go through the vtable since
  // the kernel linker can do runtime patching of vtables.
  if (Context.getLangOpts().AppleKext)
    return true;

  return !canDevirtualizeMemberFunctionCall(CE->getArg(0), MD);
}

llvm::Value *
CodeGenFunction::EmitCXXOperatorMemberCallee(const CXXOperatorCallExpr *E,
                                             const CXXMethodDecl *MD,
                                             llvm::Value *This) {
  llvm::FunctionType *fnType =
    CGM.getTypes().GetFunctionType(
                             CGM.getTypes().arrangeCXXMethodDeclaration(MD));

  if (UseVirtualCall(getContext(), E, MD))
    return BuildVirtualCall(MD, This, fnType);

  return CGM.GetAddrOfFunction(MD, fnType);
}

void CodeGenFunction::EmitForwardingCallToLambda(const CXXRecordDecl *lambda,
                                                 CallArgList &callArgs) {
  // Lookup the call operator
  DeclarationName operatorName
    = getContext().DeclarationNames.getCXXOperatorName(OO_Call);
  CXXMethodDecl *callOperator =
    cast<CXXMethodDecl>(*lambda->lookup(operatorName).first);

  // Get the address of the call operator.
  const CGFunctionInfo &calleeFnInfo =
    CGM.getTypes().arrangeCXXMethodDeclaration(callOperator);
  llvm::Value *callee =
    CGM.GetAddrOfFunction(GlobalDecl(callOperator),
                          CGM.getTypes().GetFunctionType(calleeFnInfo));

  // Prepare the return slot.
  const FunctionProtoType *FPT =
    callOperator->getType()->castAs<FunctionProtoType>();
  QualType resultType = FPT->getResultType();
  ReturnValueSlot returnSlot;
  if (!resultType->isVoidType() &&
      calleeFnInfo.getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      hasAggregateLLVMType(calleeFnInfo.getReturnType()))
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
  for (BlockDecl::param_const_iterator I = BD->param_begin(),
       E = BD->param_end(); I != E; ++I) {
    ParmVarDecl *param = *I;
    EmitDelegateCallArg(CallArgs, param);
  }

  EmitForwardingCallToLambda(Lambda, CallArgs);
}

void CodeGenFunction::EmitLambdaToBlockPointerBody(FunctionArgList &Args) {
  if (cast<CXXMethodDecl>(CurFuncDecl)->isVariadic()) {
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator forward.
    CGM.ErrorUnsupported(CurFuncDecl, "lambda conversion to variadic function");
    return;
  }

  EmitFunctionBody(Args);
}

void CodeGenFunction::EmitLambdaDelegatingInvokeBody(const CXXMethodDecl *MD) {
  const CXXRecordDecl *Lambda = MD->getParent();

  // Start building arguments for forwarding call
  CallArgList CallArgs;

  QualType ThisType = getContext().getPointerType(getContext().getRecordType(Lambda));
  llvm::Value *ThisPtr = llvm::UndefValue::get(getTypes().ConvertType(ThisType));
  CallArgs.add(RValue::get(ThisPtr), ThisType);

  // Add the rest of the parameters.
  for (FunctionDecl::param_const_iterator I = MD->param_begin(),
       E = MD->param_end(); I != E; ++I) {
    ParmVarDecl *param = *I;
    EmitDelegateCallArg(CallArgs, param);
  }

  EmitForwardingCallToLambda(Lambda, CallArgs);
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
