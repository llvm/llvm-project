//===--- CGExprConstant.cpp - Emit LLVM Code from Constant Expressions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Constant Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGObjCRuntime.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

namespace  {
class ConstStructBuilder {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;

  bool Packed;

  unsigned NextFieldOffsetInBytes;

  unsigned LLVMStructAlignment;
  
  std::vector<llvm::Constant *> Elements;

  ConstStructBuilder(CodeGenModule &CGM, CodeGenFunction *CGF)
    : CGM(CGM), CGF(CGF), Packed(false), NextFieldOffsetInBytes(0),
    LLVMStructAlignment(1) { }

  bool AppendField(const FieldDecl *Field, uint64_t FieldOffset,
                   const Expr *InitExpr) {
    uint64_t FieldOffsetInBytes = FieldOffset / 8;

    assert(NextFieldOffsetInBytes <= FieldOffsetInBytes
           && "Field offset mismatch!");

    // Emit the field.
    llvm::Constant *C = CGM.EmitConstantExpr(InitExpr, Field->getType(), CGF);
    if (!C)
      return false;

    unsigned FieldAlignment = getAlignment(C);

    // Round up the field offset to the alignment of the field type.
    uint64_t AlignedNextFieldOffsetInBytes =
      llvm::RoundUpToAlignment(NextFieldOffsetInBytes, FieldAlignment);

    if (AlignedNextFieldOffsetInBytes > FieldOffsetInBytes) {
      assert(!Packed && "Alignment is wrong even with a packed struct!");

      // Convert the struct to a packed struct.
      ConvertStructToPacked();
      
      AlignedNextFieldOffsetInBytes = NextFieldOffsetInBytes;
    }

    if (AlignedNextFieldOffsetInBytes < FieldOffsetInBytes) {
      // We need to append padding.
      AppendPadding(FieldOffsetInBytes - NextFieldOffsetInBytes);

      assert(NextFieldOffsetInBytes == FieldOffsetInBytes &&
             "Did not add enough padding!");

      AlignedNextFieldOffsetInBytes = NextFieldOffsetInBytes;
    }

    // Add the field.
    Elements.push_back(C);
    NextFieldOffsetInBytes = AlignedNextFieldOffsetInBytes + getSizeInBytes(C);
    
    if (Packed)
      assert(LLVMStructAlignment == 1 && "Packed struct not byte-aligned!");
    else
      LLVMStructAlignment = std::max(LLVMStructAlignment, FieldAlignment);

    return true;
  }

  bool AppendBitField(const FieldDecl *Field, uint64_t FieldOffset,
                      const Expr *InitExpr) {
    llvm::ConstantInt *CI =
      cast_or_null<llvm::ConstantInt>(CGM.EmitConstantExpr(InitExpr,
                                                           Field->getType(),
                                                           CGF));
    // FIXME: Can this ever happen?
    if (!CI)
      return false;

    if (FieldOffset > NextFieldOffsetInBytes * 8) {
      // We need to add padding.
      uint64_t NumBytes =
        llvm::RoundUpToAlignment(FieldOffset -
                                 NextFieldOffsetInBytes * 8, 8) / 8;

      AppendPadding(NumBytes);
    }

    uint64_t FieldSize =
      Field->getBitWidth()->EvaluateAsInt(CGM.getContext()).getZExtValue();

    llvm::APInt FieldValue = CI->getValue();

    // Promote the size of FieldValue if necessary
    // FIXME: This should never occur, but currently it can because initializer
    // constants are cast to bool, and because clang is not enforcing bitfield
    // width limits.
    if (FieldSize > FieldValue.getBitWidth())
      FieldValue.zext(FieldSize);

    // Truncate the size of FieldValue to the bit field size.
    if (FieldSize < FieldValue.getBitWidth())
      FieldValue.trunc(FieldSize);

    if (FieldOffset < NextFieldOffsetInBytes * 8) {
      // Either part of the field or the entire field can go into the previous
      // byte.
      assert(!Elements.empty() && "Elements can't be empty!");

      unsigned BitsInPreviousByte =
        NextFieldOffsetInBytes * 8 - FieldOffset;

      bool FitsCompletelyInPreviousByte =
        BitsInPreviousByte >= FieldValue.getBitWidth();

      llvm::APInt Tmp = FieldValue;

      if (!FitsCompletelyInPreviousByte) {
        unsigned NewFieldWidth = FieldSize - BitsInPreviousByte;

        if (CGM.getTargetData().isBigEndian()) {
          Tmp = Tmp.lshr(NewFieldWidth);
          Tmp.trunc(BitsInPreviousByte);

          // We want the remaining high bits.
          FieldValue.trunc(NewFieldWidth);
        } else {
          Tmp.trunc(BitsInPreviousByte);

          // We want the remaining low bits.
          FieldValue = FieldValue.lshr(BitsInPreviousByte);
          FieldValue.trunc(NewFieldWidth);
        }
      }

      Tmp.zext(8);
      if (CGM.getTargetData().isBigEndian()) {
        if (FitsCompletelyInPreviousByte)
          Tmp = Tmp.shl(BitsInPreviousByte - FieldValue.getBitWidth());
      } else {
        Tmp = Tmp.shl(8 - BitsInPreviousByte);
      }

      // Or in the bits that go into the previous byte.
      if (llvm::ConstantInt *Val = dyn_cast<llvm::ConstantInt>(Elements.back()))
        Tmp |= Val->getValue();
      else
        assert(isa<llvm::UndefValue>(Elements.back()));

      Elements.back() = llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp);

      if (FitsCompletelyInPreviousByte)
        return true;
    }

    while (FieldValue.getBitWidth() > 8) {
      llvm::APInt Tmp;

      if (CGM.getTargetData().isBigEndian()) {
        // We want the high bits.
        Tmp = FieldValue;
        Tmp = Tmp.lshr(Tmp.getBitWidth() - 8);
        Tmp.trunc(8);
      } else {
        // We want the low bits.
        Tmp = FieldValue;
        Tmp.trunc(8);

        FieldValue = FieldValue.lshr(8);
      }

      Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp));
      NextFieldOffsetInBytes++;

      FieldValue.trunc(FieldValue.getBitWidth() - 8);
    }

    assert(FieldValue.getBitWidth() > 0 &&
           "Should have at least one bit left!");
    assert(FieldValue.getBitWidth() <= 8 &&
           "Should not have more than a byte left!");

    if (FieldValue.getBitWidth() < 8) {
      if (CGM.getTargetData().isBigEndian()) {
        unsigned BitWidth = FieldValue.getBitWidth();

        FieldValue.zext(8);
        FieldValue = FieldValue << (8 - BitWidth);
      } else
        FieldValue.zext(8);
    }

    // Append the last element.
    Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(),
                                              FieldValue));
    NextFieldOffsetInBytes++;
    return true;
  }

  void AppendPadding(uint64_t NumBytes) {
    if (!NumBytes)
      return;

    const llvm::Type *Ty = llvm::Type::getInt8Ty(CGM.getLLVMContext());
    if (NumBytes > 1)
      Ty = llvm::ArrayType::get(Ty, NumBytes);

    llvm::Constant *C = llvm::Constant::getNullValue(Ty);
    Elements.push_back(C);
    assert(getAlignment(C) == 1 && "Padding must have 1 byte alignment!");

    NextFieldOffsetInBytes += getSizeInBytes(C);
  }

  void AppendTailPadding(uint64_t RecordSize) {
    assert(RecordSize % 8 == 0 && "Invalid record size!");

    uint64_t RecordSizeInBytes = RecordSize / 8;
    assert(NextFieldOffsetInBytes <= RecordSizeInBytes && "Size mismatch!");

    unsigned NumPadBytes = RecordSizeInBytes - NextFieldOffsetInBytes;
    AppendPadding(NumPadBytes);
  }

  void ConvertStructToPacked() {
    std::vector<llvm::Constant *> PackedElements;
    uint64_t ElementOffsetInBytes = 0;

    for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
      llvm::Constant *C = Elements[i];

      unsigned ElementAlign =
        CGM.getTargetData().getABITypeAlignment(C->getType());
      uint64_t AlignedElementOffsetInBytes =
        llvm::RoundUpToAlignment(ElementOffsetInBytes, ElementAlign);

      if (AlignedElementOffsetInBytes > ElementOffsetInBytes) {
        // We need some padding.
        uint64_t NumBytes =
          AlignedElementOffsetInBytes - ElementOffsetInBytes;

        const llvm::Type *Ty = llvm::Type::getInt8Ty(CGM.getLLVMContext());
        if (NumBytes > 1)
          Ty = llvm::ArrayType::get(Ty, NumBytes);

        llvm::Constant *Padding = llvm::Constant::getNullValue(Ty);
        PackedElements.push_back(Padding);
        ElementOffsetInBytes += getSizeInBytes(Padding);
      }

      PackedElements.push_back(C);
      ElementOffsetInBytes += getSizeInBytes(C);
    }

    assert(ElementOffsetInBytes == NextFieldOffsetInBytes &&
           "Packing the struct changed its size!");

    Elements = PackedElements;
    LLVMStructAlignment = 1;
    Packed = true;
  }
                              
  bool Build(InitListExpr *ILE) {
    RecordDecl *RD = ILE->getType()->getAs<RecordType>()->getDecl();
    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

    unsigned FieldNo = 0;
    unsigned ElementNo = 0;
    for (RecordDecl::field_iterator Field = RD->field_begin(),
         FieldEnd = RD->field_end();
         ElementNo < ILE->getNumInits() && Field != FieldEnd;
         ++Field, ++FieldNo) {
      if (RD->isUnion() && ILE->getInitializedFieldInUnion() != *Field)
        continue;

      if (Field->isBitField()) {
        if (!Field->getIdentifier())
          continue;

        if (!AppendBitField(*Field, Layout.getFieldOffset(FieldNo),
                            ILE->getInit(ElementNo)))
          return false;
      } else {
        if (!AppendField(*Field, Layout.getFieldOffset(FieldNo),
                         ILE->getInit(ElementNo)))
          return false;
      }

      ElementNo++;
    }

    uint64_t LayoutSizeInBytes = Layout.getSize() / 8;

    if (NextFieldOffsetInBytes > LayoutSizeInBytes) {
      // If the struct is bigger than the size of the record type,
      // we must have a flexible array member at the end.
      assert(RD->hasFlexibleArrayMember() &&
             "Must have flexible array member if struct is bigger than type!");
      
      // No tail padding is necessary.
      return true;
    }

    uint64_t LLVMSizeInBytes = llvm::RoundUpToAlignment(NextFieldOffsetInBytes, 
                                                        LLVMStructAlignment);

    // Check if we need to convert the struct to a packed struct.
    if (NextFieldOffsetInBytes <= LayoutSizeInBytes && 
        LLVMSizeInBytes > LayoutSizeInBytes) {
      assert(!Packed && "Size mismatch!");
      
      ConvertStructToPacked();
      assert(NextFieldOffsetInBytes == LayoutSizeInBytes &&
             "Converting to packed did not help!");
    }

    // Append tail padding if necessary.
    AppendTailPadding(Layout.getSize());

    assert(Layout.getSize() / 8 == NextFieldOffsetInBytes &&
           "Tail padding mismatch!");

    return true;
  }

  unsigned getAlignment(const llvm::Constant *C) const {
    if (Packed)
      return 1;

    return CGM.getTargetData().getABITypeAlignment(C->getType());
  }

  uint64_t getSizeInBytes(const llvm::Constant *C) const {
    return CGM.getTargetData().getTypeAllocSize(C->getType());
  }

public:
  static llvm::Constant *BuildStruct(CodeGenModule &CGM, CodeGenFunction *CGF,
                                     InitListExpr *ILE) {
    ConstStructBuilder Builder(CGM, CGF);

    if (!Builder.Build(ILE))
      return 0;

    llvm::Constant *Result =
      llvm::ConstantStruct::get(CGM.getLLVMContext(),
                                Builder.Elements, Builder.Packed);

    assert(llvm::RoundUpToAlignment(Builder.NextFieldOffsetInBytes,
                                    Builder.getAlignment(Result)) ==
           Builder.getSizeInBytes(Result) && "Size mismatch!");

    return Result;
  }
};

class ConstExprEmitter :
  public StmtVisitor<ConstExprEmitter, llvm::Constant*> {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;
  llvm::LLVMContext &VMContext;
public:
  ConstExprEmitter(CodeGenModule &cgm, CodeGenFunction *cgf)
    : CGM(cgm), CGF(cgf), VMContext(cgm.getLLVMContext()) {
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  llvm::Constant *VisitStmt(Stmt *S) {
    return 0;
  }

  llvm::Constant *VisitParenExpr(ParenExpr *PE) {
    return Visit(PE->getSubExpr());
  }

  llvm::Constant *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return Visit(E->getInitializer());
  }
    
  llvm::Constant *EmitMemberFunctionPointer(CXXMethodDecl *MD) {
    assert(MD->isInstance() && "Member function must not be static!");
    
    MD = MD->getCanonicalDecl();

    const llvm::Type *PtrDiffTy = 
      CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());
    
    llvm::Constant *Values[2];
    
    // Get the function pointer (or index if this is a virtual function).
    if (MD->isVirtual()) {
      uint64_t Index = CGM.getVtableInfo().getMethodVtableIndex(MD);

      // Itanium C++ ABI 2.3:
      //   For a non-virtual function, this field is a simple function pointer. 
      //   For a virtual function, it is 1 plus the virtual table offset 
      //   (in bytes) of the function, represented as a ptrdiff_t. 
      Values[0] = llvm::ConstantInt::get(PtrDiffTy, (Index * 8) + 1);
    } else {
      const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
      const llvm::Type *Ty =
        CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                       FPT->isVariadic());

      llvm::Constant *FuncPtr = CGM.GetAddrOfFunction(MD, Ty);
      Values[0] = llvm::ConstantExpr::getPtrToInt(FuncPtr, PtrDiffTy);
    } 
    
    // The adjustment will always be 0.
    Values[1] = llvm::ConstantInt::get(PtrDiffTy, 0);
    
    return llvm::ConstantStruct::get(CGM.getLLVMContext(),
                                     Values, 2, /*Packed=*/false);
  }

  llvm::Constant *VisitUnaryAddrOf(UnaryOperator *E) {
    if (const MemberPointerType *MPT = 
        E->getType()->getAs<MemberPointerType>()) {
      QualType T = MPT->getPointeeType();
      DeclRefExpr *DRE = cast<DeclRefExpr>(E->getSubExpr());

      NamedDecl *ND = DRE->getDecl();
      if (T->isFunctionProtoType())
        return EmitMemberFunctionPointer(cast<CXXMethodDecl>(ND));
      
      // We have a pointer to data member.
      return CGM.EmitPointerToDataMember(cast<FieldDecl>(ND));
    }

    return 0;
  }
    
  llvm::Constant *VisitBinSub(BinaryOperator *E) {
    // This must be a pointer/pointer subtraction.  This only happens for
    // address of label.
    if (!isa<AddrLabelExpr>(E->getLHS()->IgnoreParenNoopCasts(CGM.getContext())) ||
       !isa<AddrLabelExpr>(E->getRHS()->IgnoreParenNoopCasts(CGM.getContext())))
      return 0;
    
    llvm::Constant *LHS = CGM.EmitConstantExpr(E->getLHS(),
                                               E->getLHS()->getType(), CGF);
    llvm::Constant *RHS = CGM.EmitConstantExpr(E->getRHS(),
                                               E->getRHS()->getType(), CGF);

    const llvm::Type *ResultType = ConvertType(E->getType());
    LHS = llvm::ConstantExpr::getPtrToInt(LHS, ResultType);
    RHS = llvm::ConstantExpr::getPtrToInt(RHS, ResultType);
        
    // No need to divide by element size, since addr of label is always void*,
    // which has size 1 in GNUish.
    return llvm::ConstantExpr::getSub(LHS, RHS);
  }
    
  llvm::Constant *VisitCastExpr(CastExpr* E) {
    switch (E->getCastKind()) {
    case CastExpr::CK_ToUnion: {
      // GCC cast to union extension
      assert(E->getType()->isUnionType() &&
             "Destination type is not union type!");
      const llvm::Type *Ty = ConvertType(E->getType());
      Expr *SubExpr = E->getSubExpr();

      llvm::Constant *C =
        CGM.EmitConstantExpr(SubExpr, SubExpr->getType(), CGF);
      if (!C)
        return 0;

      // Build a struct with the union sub-element as the first member,
      // and padded to the appropriate size
      std::vector<llvm::Constant*> Elts;
      std::vector<const llvm::Type*> Types;
      Elts.push_back(C);
      Types.push_back(C->getType());
      unsigned CurSize = CGM.getTargetData().getTypeAllocSize(C->getType());
      unsigned TotalSize = CGM.getTargetData().getTypeAllocSize(Ty);

      assert(CurSize <= TotalSize && "Union size mismatch!");
      if (unsigned NumPadBytes = TotalSize - CurSize) {
        const llvm::Type *Ty = llvm::Type::getInt8Ty(VMContext);
        if (NumPadBytes > 1)
          Ty = llvm::ArrayType::get(Ty, NumPadBytes);

        Elts.push_back(llvm::Constant::getNullValue(Ty));
        Types.push_back(Ty);
      }

      llvm::StructType* STy =
        llvm::StructType::get(C->getType()->getContext(), Types, false);
      return llvm::ConstantStruct::get(STy, Elts);
    }
    case CastExpr::CK_NullToMemberPointer:
      return CGM.EmitNullConstant(E->getType());
      
    case CastExpr::CK_BaseToDerivedMemberPointer: {
      Expr *SubExpr = E->getSubExpr();

      const MemberPointerType *SrcTy = 
        SubExpr->getType()->getAs<MemberPointerType>();
      const MemberPointerType *DestTy = 
        E->getType()->getAs<MemberPointerType>();
      
      const CXXRecordDecl *BaseClass =
        cast<CXXRecordDecl>(cast<RecordType>(SrcTy->getClass())->getDecl());
      const CXXRecordDecl *DerivedClass =
        cast<CXXRecordDecl>(cast<RecordType>(DestTy->getClass())->getDecl());

      if (SrcTy->getPointeeType()->isFunctionProtoType()) {
        llvm::Constant *C = 
          CGM.EmitConstantExpr(SubExpr, SubExpr->getType(), CGF);
        if (!C)
          return 0;
        
        llvm::ConstantStruct *CS = cast<llvm::ConstantStruct>(C);
        
        // Check if we need to update the adjustment.
        if (llvm::Constant *Offset = 
              CGM.GetNonVirtualBaseClassOffset(DerivedClass, BaseClass)) {
          llvm::Constant *Values[2];
        
          Values[0] = CS->getOperand(0);
          Values[1] = llvm::ConstantExpr::getAdd(CS->getOperand(1), Offset);
          return llvm::ConstantStruct::get(CGM.getLLVMContext(), Values, 2, 
                                           /*Packed=*/false);
        }
        
        return CS;
      }          
    }

    case CastExpr::CK_BitCast: 
      // This must be a member function pointer cast.
      return Visit(E->getSubExpr());

    default: {
      // FIXME: This should be handled by the CK_NoOp cast kind.
      // Explicit and implicit no-op casts
      QualType Ty = E->getType(), SubTy = E->getSubExpr()->getType();
      if (CGM.getContext().hasSameUnqualifiedType(Ty, SubTy))
        return Visit(E->getSubExpr());

      // Handle integer->integer casts for address-of-label differences.
      if (Ty->isIntegerType() && SubTy->isIntegerType() &&
          CGF) {
        llvm::Value *Src = Visit(E->getSubExpr());
        if (Src == 0) return 0;
        
        // Use EmitScalarConversion to perform the conversion.
        return cast<llvm::Constant>(CGF->EmitScalarConversion(Src, SubTy, Ty));
      }
      
      return 0;
    }
    }
  }

  llvm::Constant *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }

  llvm::Constant *EmitArrayInitialization(InitListExpr *ILE) {
    std::vector<llvm::Constant*> Elts;
    const llvm::ArrayType *AType =
        cast<llvm::ArrayType>(ConvertType(ILE->getType()));
    unsigned NumInitElements = ILE->getNumInits();
    // FIXME: Check for wide strings
    // FIXME: Check for NumInitElements exactly equal to 1??
    if (NumInitElements > 0 &&
        (isa<StringLiteral>(ILE->getInit(0)) ||
         isa<ObjCEncodeExpr>(ILE->getInit(0))) &&
        ILE->getType()->getArrayElementTypeNoTypeQual()->isCharType())
      return Visit(ILE->getInit(0));
    const llvm::Type *ElemTy = AType->getElementType();
    unsigned NumElements = AType->getNumElements();

    // Initialising an array requires us to automatically
    // initialise any elements that have not been initialised explicitly
    unsigned NumInitableElts = std::min(NumInitElements, NumElements);

    // Copy initializer elements.
    unsigned i = 0;
    bool RewriteType = false;
    for (; i < NumInitableElts; ++i) {
      Expr *Init = ILE->getInit(i);
      llvm::Constant *C = CGM.EmitConstantExpr(Init, Init->getType(), CGF);
      if (!C)
        return 0;
      RewriteType |= (C->getType() != ElemTy);
      Elts.push_back(C);
    }

    // Initialize remaining array elements.
    // FIXME: This doesn't handle member pointers correctly!
    for (; i < NumElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(ElemTy));

    if (RewriteType) {
      // FIXME: Try to avoid packing the array
      std::vector<const llvm::Type*> Types;
      for (unsigned i = 0; i < Elts.size(); ++i)
        Types.push_back(Elts[i]->getType());
      const llvm::StructType *SType = llvm::StructType::get(AType->getContext(),
                                                            Types, true);
      return llvm::ConstantStruct::get(SType, Elts);
    }

    return llvm::ConstantArray::get(AType, Elts);
  }

  llvm::Constant *EmitStructInitialization(InitListExpr *ILE) {
    return ConstStructBuilder::BuildStruct(CGM, CGF, ILE);
  }

  llvm::Constant *EmitUnionInitialization(InitListExpr *ILE) {
    return ConstStructBuilder::BuildStruct(CGM, CGF, ILE);
  }

  llvm::Constant *VisitImplicitValueInitExpr(ImplicitValueInitExpr* E) {
    return CGM.EmitNullConstant(E->getType());
  }

  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    if (ILE->getType()->isScalarType()) {
      // We have a scalar in braces. Just use the first element.
      if (ILE->getNumInits() > 0) {
        Expr *Init = ILE->getInit(0);
        return CGM.EmitConstantExpr(Init, Init->getType(), CGF);
      }
      return CGM.EmitNullConstant(ILE->getType());
    }

    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE);

    if (ILE->getType()->isRecordType())
      return EmitStructInitialization(ILE);

    if (ILE->getType()->isUnionType())
      return EmitUnionInitialization(ILE);

    // If ILE was a constant vector, we would have handled it already.
    if (ILE->getType()->isVectorType())
      return 0;

    assert(0 && "Unable to handle InitListExpr");
    // Get rid of control reaches end of void function warning.
    // Not reached.
    return 0;
  }

  llvm::Constant *VisitCXXConstructExpr(CXXConstructExpr *E) {
    if (!E->getConstructor()->isTrivial())
      return 0;

    QualType Ty = E->getType();

    // FIXME: We should not have to call getBaseElementType here.
    const RecordType *RT = 
      CGM.getContext().getBaseElementType(Ty)->getAs<RecordType>();
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    
    // If the class doesn't have a trivial destructor, we can't emit it as a
    // constant expr.
    if (!RD->hasTrivialDestructor())
      return 0;
    
    // Only copy and default constructors can be trivial.


    if (E->getNumArgs()) {
      assert(E->getNumArgs() == 1 && "trivial ctor with > 1 argument");
      assert(E->getConstructor()->isCopyConstructor() &&
             "trivial ctor has argument but isn't a copy ctor");

      Expr *Arg = E->getArg(0);
      assert(CGM.getContext().hasSameUnqualifiedType(Ty, Arg->getType()) &&
             "argument to copy ctor is of wrong type");

      return Visit(Arg);
    }

    return CGM.EmitNullConstant(Ty);
  }

  llvm::Constant *VisitStringLiteral(StringLiteral *E) {
    assert(!E->getType()->isPointerType() && "Strings are always arrays");

    // This must be a string initializing an array in a static initializer.
    // Don't emit it as the address of the string, emit the string data itself
    // as an inline array.
    return llvm::ConstantArray::get(VMContext,
                                    CGM.GetStringForStringLiteral(E), false);
  }

  llvm::Constant *VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
    // This must be an @encode initializing an array in a static initializer.
    // Don't emit it as the address of the string, emit the string data itself
    // as an inline array.
    std::string Str;
    CGM.getContext().getObjCEncodingForType(E->getEncodedType(), Str);
    const ConstantArrayType *CAT = cast<ConstantArrayType>(E->getType());

    // Resize the string to the right size, adding zeros at the end, or
    // truncating as needed.
    Str.resize(CAT->getSize().getZExtValue(), '\0');
    return llvm::ConstantArray::get(VMContext, Str, false);
  }

  llvm::Constant *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }

  // Utility methods
  const llvm::Type *ConvertType(QualType T) {
    return CGM.getTypes().ConvertType(T);
  }

public:
  llvm::Constant *EmitLValue(Expr *E) {
    switch (E->getStmtClass()) {
    default: break;
    case Expr::CompoundLiteralExprClass: {
      // Note that due to the nature of compound literals, this is guaranteed
      // to be the only use of the variable, so we just generate it here.
      CompoundLiteralExpr *CLE = cast<CompoundLiteralExpr>(E);
      llvm::Constant* C = Visit(CLE->getInitializer());
      // FIXME: "Leaked" on failure.
      if (C)
        C = new llvm::GlobalVariable(CGM.getModule(), C->getType(),
                                     E->getType().isConstant(CGM.getContext()),
                                     llvm::GlobalValue::InternalLinkage,
                                     C, ".compoundliteral", 0, false,
                                     E->getType().getAddressSpace());
      return C;
    }
    case Expr::DeclRefExprClass: {
      ValueDecl *Decl = cast<DeclRefExpr>(E)->getDecl();
      if (Decl->hasAttr<WeakRefAttr>())
	return CGM.GetWeakRefReference(Decl);
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Decl))
        return CGM.GetAddrOfFunction(FD);
      if (const VarDecl* VD = dyn_cast<VarDecl>(Decl)) {
        // We can never refer to a variable with local storage.
        if (!VD->hasLocalStorage()) {
          if (VD->isFileVarDecl() || VD->hasExternalStorage())
            return CGM.GetAddrOfGlobalVar(VD);
          else if (VD->isBlockVarDecl()) {
            assert(CGF && "Can't access static local vars without CGF");
            return CGF->GetAddrOfStaticLocalVar(VD);
          }
        }
      }
      break;
    }
    case Expr::StringLiteralClass:
      return CGM.GetAddrOfConstantStringFromLiteral(cast<StringLiteral>(E));
    case Expr::ObjCEncodeExprClass:
      return CGM.GetAddrOfConstantStringFromObjCEncode(cast<ObjCEncodeExpr>(E));
    case Expr::ObjCStringLiteralClass: {
      ObjCStringLiteral* SL = cast<ObjCStringLiteral>(E);
      llvm::Constant *C =
          CGM.getObjCRuntime().GenerateConstantString(SL->getString());
      return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
    }
    case Expr::PredefinedExprClass: {
      unsigned Type = cast<PredefinedExpr>(E)->getIdentType();
      if (CGF) {
        LValue Res = CGF->EmitPredefinedFunctionName(Type);
        return cast<llvm::Constant>(Res.getAddress());
      } else if (Type == PredefinedExpr::PrettyFunction) {
        return CGM.GetAddrOfConstantCString("top level", ".tmp");
      }

      return CGM.GetAddrOfConstantCString("", ".tmp");
    }
    case Expr::AddrLabelExprClass: {
      assert(CGF && "Invalid address of label expression outside function.");
      llvm::Constant *Ptr =
        CGF->GetAddrOfLabel(cast<AddrLabelExpr>(E)->getLabel());
      return llvm::ConstantExpr::getBitCast(Ptr, ConvertType(E->getType()));
    }
    case Expr::CallExprClass: {
      CallExpr* CE = cast<CallExpr>(E);
      unsigned builtin = CE->isBuiltinCall(CGM.getContext());
      if (builtin !=
            Builtin::BI__builtin___CFStringMakeConstantString &&
          builtin !=
            Builtin::BI__builtin___NSStringMakeConstantString)
        break;
      const Expr *Arg = CE->getArg(0)->IgnoreParenCasts();
      const StringLiteral *Literal = cast<StringLiteral>(Arg);
      if (builtin ==
            Builtin::BI__builtin___NSStringMakeConstantString) {
        return CGM.getObjCRuntime().GenerateConstantString(Literal);
      }
      // FIXME: need to deal with UCN conversion issues.
      return CGM.GetAddrOfConstantCFString(Literal);
    }
    case Expr::BlockExprClass: {
      std::string FunctionName;
      if (CGF)
        FunctionName = CGF->CurFn->getName();
      else
        FunctionName = "global";

      return CGM.GetAddrOfGlobalBlock(cast<BlockExpr>(E), FunctionName.c_str());
    }
    }

    return 0;
  }
};

}  // end anonymous namespace.

llvm::Constant *CodeGenModule::EmitConstantExpr(const Expr *E,
                                                QualType DestType,
                                                CodeGenFunction *CGF) {
  Expr::EvalResult Result;

  bool Success = false;

  if (DestType->isReferenceType())
    Success = E->EvaluateAsLValue(Result, Context);
  else
    Success = E->Evaluate(Result, Context);

  if (Success && !Result.HasSideEffects) {
    switch (Result.Val.getKind()) {
    case APValue::Uninitialized:
      assert(0 && "Constant expressions should be initialized.");
      return 0;
    case APValue::LValue: {
      const llvm::Type *DestTy = getTypes().ConvertTypeForMem(DestType);
      llvm::Constant *Offset =
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                               Result.Val.getLValueOffset().getQuantity());

      llvm::Constant *C;
      if (const Expr *LVBase = Result.Val.getLValueBase()) {
        C = ConstExprEmitter(*this, CGF).EmitLValue(const_cast<Expr*>(LVBase));

        // Apply offset if necessary.
        if (!Offset->isNullValue()) {
          const llvm::Type *Type = llvm::Type::getInt8PtrTy(VMContext);
          llvm::Constant *Casted = llvm::ConstantExpr::getBitCast(C, Type);
          Casted = llvm::ConstantExpr::getGetElementPtr(Casted, &Offset, 1);
          C = llvm::ConstantExpr::getBitCast(Casted, C->getType());
        }

        // Convert to the appropriate type; this could be an lvalue for
        // an integer.
        if (isa<llvm::PointerType>(DestTy))
          return llvm::ConstantExpr::getBitCast(C, DestTy);

        return llvm::ConstantExpr::getPtrToInt(C, DestTy);
      } else {
        C = Offset;

        // Convert to the appropriate type; this could be an lvalue for
        // an integer.
        if (isa<llvm::PointerType>(DestTy))
          return llvm::ConstantExpr::getIntToPtr(C, DestTy);

        // If the types don't match this should only be a truncate.
        if (C->getType() != DestTy)
          return llvm::ConstantExpr::getTrunc(C, DestTy);

        return C;
      }
    }
    case APValue::Int: {
      llvm::Constant *C = llvm::ConstantInt::get(VMContext,
                                                 Result.Val.getInt());

      if (C->getType() == llvm::Type::getInt1Ty(VMContext)) {
        const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
        C = llvm::ConstantExpr::getZExt(C, BoolTy);
      }
      return C;
    }
    case APValue::ComplexInt: {
      llvm::Constant *Complex[2];

      Complex[0] = llvm::ConstantInt::get(VMContext,
                                          Result.Val.getComplexIntReal());
      Complex[1] = llvm::ConstantInt::get(VMContext,
                                          Result.Val.getComplexIntImag());

      // FIXME: the target may want to specify that this is packed.
      return llvm::ConstantStruct::get(VMContext, Complex, 2, false);
    }
    case APValue::Float:
      return llvm::ConstantFP::get(VMContext, Result.Val.getFloat());
    case APValue::ComplexFloat: {
      llvm::Constant *Complex[2];

      Complex[0] = llvm::ConstantFP::get(VMContext,
                                         Result.Val.getComplexFloatReal());
      Complex[1] = llvm::ConstantFP::get(VMContext,
                                         Result.Val.getComplexFloatImag());

      // FIXME: the target may want to specify that this is packed.
      return llvm::ConstantStruct::get(VMContext, Complex, 2, false);
    }
    case APValue::Vector: {
      llvm::SmallVector<llvm::Constant *, 4> Inits;
      unsigned NumElts = Result.Val.getVectorLength();

      for (unsigned i = 0; i != NumElts; ++i) {
        APValue &Elt = Result.Val.getVectorElt(i);
        if (Elt.isInt())
          Inits.push_back(llvm::ConstantInt::get(VMContext, Elt.getInt()));
        else
          Inits.push_back(llvm::ConstantFP::get(VMContext, Elt.getFloat()));
      }
      return llvm::ConstantVector::get(&Inits[0], Inits.size());
    }
    }
  }

  llvm::Constant* C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
  if (C && C->getType() == llvm::Type::getInt1Ty(VMContext)) {
    const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}

static bool containsPointerToDataMember(CodeGenTypes &Types, QualType T) {
  // No need to check for member pointers when not compiling C++.
  if (!Types.getContext().getLangOptions().CPlusPlus)
    return false;
  
  T = Types.getContext().getBaseElementType(T);
  
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    
    // FIXME: It would be better if there was a way to explicitly compute the
    // record layout instead of converting to a type.
    Types.ConvertTagDeclType(RD);
    
    const CGRecordLayout &Layout = Types.getCGRecordLayout(RD);
    return Layout.containsPointerToDataMember();
  }
    
  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>())
    return !MPT->getPointeeType()->isFunctionType();
  
  return false;
}

llvm::Constant *CodeGenModule::EmitNullConstant(QualType T) {
  if (!containsPointerToDataMember(getTypes(), T))
    return llvm::Constant::getNullValue(getTypes().ConvertTypeForMem(T));
    
  if (const ConstantArrayType *CAT = Context.getAsConstantArrayType(T)) {

    QualType ElementTy = CAT->getElementType();

    llvm::Constant *Element = EmitNullConstant(ElementTy);
    unsigned NumElements = CAT->getSize().getZExtValue();
    std::vector<llvm::Constant *> Array(NumElements);
    for (unsigned i = 0; i != NumElements; ++i)
      Array[i] = Element;

    const llvm::ArrayType *ATy =
      cast<llvm::ArrayType>(getTypes().ConvertTypeForMem(T));
    return llvm::ConstantArray::get(ATy, Array);
  }

  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    assert(!RD->getNumBases() && 
           "FIXME: Handle zero-initializing structs with bases and "
           "pointers to data members.");
    const llvm::StructType *STy =
      cast<llvm::StructType>(getTypes().ConvertTypeForMem(T));
    unsigned NumElements = STy->getNumElements();
    std::vector<llvm::Constant *> Elements(NumElements);

    for (RecordDecl::field_iterator I = RD->field_begin(),
         E = RD->field_end(); I != E; ++I) {
      const FieldDecl *FD = *I;
      
      unsigned FieldNo = getTypes().getLLVMFieldNo(FD);
      Elements[FieldNo] = EmitNullConstant(FD->getType());
    }
    
    // Now go through all other fields and zero them out.
    for (unsigned i = 0; i != NumElements; ++i) {
      if (!Elements[i])
        Elements[i] = llvm::Constant::getNullValue(STy->getElementType(i));
    }
    
    return llvm::ConstantStruct::get(STy, Elements);
  }

  assert(!T->getAs<MemberPointerType>()->getPointeeType()->isFunctionType() &&
         "Should only see pointers to data members here!");
  
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  return llvm::ConstantInt::get(getTypes().ConvertTypeForMem(T), -1ULL, 
                                /*isSigned=*/true);
}

llvm::Constant *
CodeGenModule::EmitPointerToDataMember(const FieldDecl *FD) {

  // Itanium C++ ABI 2.3:
  //   A pointer to data member is an offset from the base address of the class
  //   object containing it, represented as a ptrdiff_t

  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(FD->getParent());
  QualType ClassType = 
    getContext().getTypeDeclType(const_cast<CXXRecordDecl *>(ClassDecl));
  
  const llvm::StructType *ClassLTy =
    cast<llvm::StructType>(getTypes().ConvertType(ClassType));

  unsigned FieldNo = getTypes().getLLVMFieldNo(FD);
  uint64_t Offset = 
    getTargetData().getStructLayout(ClassLTy)->getElementOffset(FieldNo);

  const llvm::Type *PtrDiffTy = 
    getTypes().ConvertType(getContext().getPointerDiffType());

  return llvm::ConstantInt::get(PtrDiffTy, Offset);
}
