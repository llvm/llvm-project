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
#include "CGCXXABI.h"
#include "CGObjCRuntime.h"
#include "CGRecordLayout.h"
#include "CodeGenModule.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                            ConstStructBuilder
//===----------------------------------------------------------------------===//

namespace {
class ConstStructBuilder {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;

  bool Packed;
  CharUnits NextFieldOffsetInChars;
  CharUnits LLVMStructAlignment;
  SmallVector<llvm::Constant *, 32> Elements;
public:
  static llvm::Constant *BuildStruct(CodeGenModule &CGM, CodeGenFunction *CGF,
                                     InitListExpr *ILE);
  static llvm::Constant *BuildStruct(CodeGenModule &CGM, CodeGenFunction *CGF,
                                     const APValue &Value, QualType ValTy);

private:
  ConstStructBuilder(CodeGenModule &CGM, CodeGenFunction *CGF)
    : CGM(CGM), CGF(CGF), Packed(false), 
    NextFieldOffsetInChars(CharUnits::Zero()),
    LLVMStructAlignment(CharUnits::One()) { }

  void AppendField(const FieldDecl *Field, uint64_t FieldOffset,
                   llvm::Constant *InitExpr);

  void AppendBytes(CharUnits FieldOffsetInChars, llvm::Constant *InitCst);

  void AppendBitField(const FieldDecl *Field, uint64_t FieldOffset,
                      llvm::ConstantInt *InitExpr);

  void AppendPadding(CharUnits PadSize);

  void AppendTailPadding(CharUnits RecordSize);

  void ConvertStructToPacked();

  bool Build(InitListExpr *ILE);
  void Build(const APValue &Val, const RecordDecl *RD, bool IsPrimaryBase,
             const CXXRecordDecl *VTableClass, CharUnits BaseOffset);
  llvm::Constant *Finalize(QualType Ty);

  CharUnits getAlignment(const llvm::Constant *C) const {
    if (Packed)  return CharUnits::One();
    return CharUnits::fromQuantity(
        CGM.getDataLayout().getABITypeAlignment(C->getType()));
  }

  CharUnits getSizeInChars(const llvm::Constant *C) const {
    return CharUnits::fromQuantity(
        CGM.getDataLayout().getTypeAllocSize(C->getType()));
  }
};

void ConstStructBuilder::
AppendField(const FieldDecl *Field, uint64_t FieldOffset,
            llvm::Constant *InitCst) {
  const ASTContext &Context = CGM.getContext();

  CharUnits FieldOffsetInChars = Context.toCharUnitsFromBits(FieldOffset);

  AppendBytes(FieldOffsetInChars, InitCst);
}

void ConstStructBuilder::
AppendBytes(CharUnits FieldOffsetInChars, llvm::Constant *InitCst) {

  assert(NextFieldOffsetInChars <= FieldOffsetInChars
         && "Field offset mismatch!");

  CharUnits FieldAlignment = getAlignment(InitCst);

  // Round up the field offset to the alignment of the field type.
  CharUnits AlignedNextFieldOffsetInChars =
    NextFieldOffsetInChars.RoundUpToAlignment(FieldAlignment);

  if (AlignedNextFieldOffsetInChars > FieldOffsetInChars) {
    assert(!Packed && "Alignment is wrong even with a packed struct!");

    // Convert the struct to a packed struct.
    ConvertStructToPacked();

    AlignedNextFieldOffsetInChars = NextFieldOffsetInChars;
  }

  if (AlignedNextFieldOffsetInChars < FieldOffsetInChars) {
    // We need to append padding.
    AppendPadding(FieldOffsetInChars - NextFieldOffsetInChars);

    assert(NextFieldOffsetInChars == FieldOffsetInChars &&
           "Did not add enough padding!");

    AlignedNextFieldOffsetInChars = NextFieldOffsetInChars;
  }

  // Add the field.
  Elements.push_back(InitCst);
  NextFieldOffsetInChars = AlignedNextFieldOffsetInChars +
                           getSizeInChars(InitCst);

  if (Packed)
    assert(LLVMStructAlignment == CharUnits::One() &&
           "Packed struct not byte-aligned!");
  else
    LLVMStructAlignment = std::max(LLVMStructAlignment, FieldAlignment);
}

void ConstStructBuilder::AppendBitField(const FieldDecl *Field,
                                        uint64_t FieldOffset,
                                        llvm::ConstantInt *CI) {
  const ASTContext &Context = CGM.getContext();
  const uint64_t CharWidth = Context.getCharWidth();
  uint64_t NextFieldOffsetInBits = Context.toBits(NextFieldOffsetInChars);
  if (FieldOffset > NextFieldOffsetInBits) {
    // We need to add padding.
    CharUnits PadSize = Context.toCharUnitsFromBits(
      llvm::RoundUpToAlignment(FieldOffset - NextFieldOffsetInBits, 
                               Context.getTargetInfo().getCharAlign()));

    AppendPadding(PadSize);
  }

  uint64_t FieldSize = Field->getBitWidthValue(Context);

  llvm::APInt FieldValue = CI->getValue();

  // Promote the size of FieldValue if necessary
  // FIXME: This should never occur, but currently it can because initializer
  // constants are cast to bool, and because clang is not enforcing bitfield
  // width limits.
  if (FieldSize > FieldValue.getBitWidth())
    FieldValue = FieldValue.zext(FieldSize);

  // Truncate the size of FieldValue to the bit field size.
  if (FieldSize < FieldValue.getBitWidth())
    FieldValue = FieldValue.trunc(FieldSize);

  NextFieldOffsetInBits = Context.toBits(NextFieldOffsetInChars);
  if (FieldOffset < NextFieldOffsetInBits) {
    // Either part of the field or the entire field can go into the previous
    // byte.
    assert(!Elements.empty() && "Elements can't be empty!");

    unsigned BitsInPreviousByte = NextFieldOffsetInBits - FieldOffset;

    bool FitsCompletelyInPreviousByte =
      BitsInPreviousByte >= FieldValue.getBitWidth();

    llvm::APInt Tmp = FieldValue;

    if (!FitsCompletelyInPreviousByte) {
      unsigned NewFieldWidth = FieldSize - BitsInPreviousByte;

      if (CGM.getDataLayout().isBigEndian()) {
        Tmp = Tmp.lshr(NewFieldWidth);
        Tmp = Tmp.trunc(BitsInPreviousByte);

        // We want the remaining high bits.
        FieldValue = FieldValue.trunc(NewFieldWidth);
      } else {
        Tmp = Tmp.trunc(BitsInPreviousByte);

        // We want the remaining low bits.
        FieldValue = FieldValue.lshr(BitsInPreviousByte);
        FieldValue = FieldValue.trunc(NewFieldWidth);
      }
    }

    Tmp = Tmp.zext(CharWidth);
    if (CGM.getDataLayout().isBigEndian()) {
      if (FitsCompletelyInPreviousByte)
        Tmp = Tmp.shl(BitsInPreviousByte - FieldValue.getBitWidth());
    } else {
      Tmp = Tmp.shl(CharWidth - BitsInPreviousByte);
    }

    // 'or' in the bits that go into the previous byte.
    llvm::Value *LastElt = Elements.back();
    if (llvm::ConstantInt *Val = dyn_cast<llvm::ConstantInt>(LastElt))
      Tmp |= Val->getValue();
    else {
      assert(isa<llvm::UndefValue>(LastElt));
      // If there is an undef field that we're adding to, it can either be a
      // scalar undef (in which case, we just replace it with our field) or it
      // is an array.  If it is an array, we have to pull one byte off the
      // array so that the other undef bytes stay around.
      if (!isa<llvm::IntegerType>(LastElt->getType())) {
        // The undef padding will be a multibyte array, create a new smaller
        // padding and then an hole for our i8 to get plopped into.
        assert(isa<llvm::ArrayType>(LastElt->getType()) &&
               "Expected array padding of undefs");
        llvm::ArrayType *AT = cast<llvm::ArrayType>(LastElt->getType());
        assert(AT->getElementType()->isIntegerTy(CharWidth) &&
               AT->getNumElements() != 0 &&
               "Expected non-empty array padding of undefs");
        
        // Remove the padding array.
        NextFieldOffsetInChars -= CharUnits::fromQuantity(AT->getNumElements());
        Elements.pop_back();
        
        // Add the padding back in two chunks.
        AppendPadding(CharUnits::fromQuantity(AT->getNumElements()-1));
        AppendPadding(CharUnits::One());
        assert(isa<llvm::UndefValue>(Elements.back()) &&
               Elements.back()->getType()->isIntegerTy(CharWidth) &&
               "Padding addition didn't work right");
      }
    }

    Elements.back() = llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp);

    if (FitsCompletelyInPreviousByte)
      return;
  }

  while (FieldValue.getBitWidth() > CharWidth) {
    llvm::APInt Tmp;

    if (CGM.getDataLayout().isBigEndian()) {
      // We want the high bits.
      Tmp = 
        FieldValue.lshr(FieldValue.getBitWidth() - CharWidth).trunc(CharWidth);
    } else {
      // We want the low bits.
      Tmp = FieldValue.trunc(CharWidth);

      FieldValue = FieldValue.lshr(CharWidth);
    }

    Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp));
    ++NextFieldOffsetInChars;

    FieldValue = FieldValue.trunc(FieldValue.getBitWidth() - CharWidth);
  }

  assert(FieldValue.getBitWidth() > 0 &&
         "Should have at least one bit left!");
  assert(FieldValue.getBitWidth() <= CharWidth &&
         "Should not have more than a byte left!");

  if (FieldValue.getBitWidth() < CharWidth) {
    if (CGM.getDataLayout().isBigEndian()) {
      unsigned BitWidth = FieldValue.getBitWidth();

      FieldValue = FieldValue.zext(CharWidth) << (CharWidth - BitWidth);
    } else
      FieldValue = FieldValue.zext(CharWidth);
  }

  // Append the last element.
  Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(),
                                            FieldValue));
  ++NextFieldOffsetInChars;
}

void ConstStructBuilder::AppendPadding(CharUnits PadSize) {
  if (PadSize.isZero())
    return;

  llvm::Type *Ty = CGM.Int8Ty;
  if (PadSize > CharUnits::One())
    Ty = llvm::ArrayType::get(Ty, PadSize.getQuantity());

  llvm::Constant *C = llvm::UndefValue::get(Ty);
  Elements.push_back(C);
  assert(getAlignment(C) == CharUnits::One() && 
         "Padding must have 1 byte alignment!");

  NextFieldOffsetInChars += getSizeInChars(C);
}

void ConstStructBuilder::AppendTailPadding(CharUnits RecordSize) {
  assert(NextFieldOffsetInChars <= RecordSize && 
         "Size mismatch!");

  AppendPadding(RecordSize - NextFieldOffsetInChars);
}

void ConstStructBuilder::ConvertStructToPacked() {
  SmallVector<llvm::Constant *, 16> PackedElements;
  CharUnits ElementOffsetInChars = CharUnits::Zero();

  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    llvm::Constant *C = Elements[i];

    CharUnits ElementAlign = CharUnits::fromQuantity(
      CGM.getDataLayout().getABITypeAlignment(C->getType()));
    CharUnits AlignedElementOffsetInChars =
      ElementOffsetInChars.RoundUpToAlignment(ElementAlign);

    if (AlignedElementOffsetInChars > ElementOffsetInChars) {
      // We need some padding.
      CharUnits NumChars =
        AlignedElementOffsetInChars - ElementOffsetInChars;

      llvm::Type *Ty = CGM.Int8Ty;
      if (NumChars > CharUnits::One())
        Ty = llvm::ArrayType::get(Ty, NumChars.getQuantity());

      llvm::Constant *Padding = llvm::UndefValue::get(Ty);
      PackedElements.push_back(Padding);
      ElementOffsetInChars += getSizeInChars(Padding);
    }

    PackedElements.push_back(C);
    ElementOffsetInChars += getSizeInChars(C);
  }

  assert(ElementOffsetInChars == NextFieldOffsetInChars &&
         "Packing the struct changed its size!");

  Elements.swap(PackedElements);
  LLVMStructAlignment = CharUnits::One();
  Packed = true;
}
                            
bool ConstStructBuilder::Build(InitListExpr *ILE) {
  RecordDecl *RD = ILE->getType()->getAs<RecordType>()->getDecl();
  const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

  unsigned FieldNo = 0;
  unsigned ElementNo = 0;
  
  for (RecordDecl::field_iterator Field = RD->field_begin(),
       FieldEnd = RD->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    // If this is a union, skip all the fields that aren't being initialized.
    if (RD->isUnion() && ILE->getInitializedFieldInUnion() != *Field)
      continue;

    // Don't emit anonymous bitfields, they just affect layout.
    if (Field->isUnnamedBitfield())
      continue;

    // Get the initializer.  A struct can include fields without initializers,
    // we just use explicit null values for them.
    llvm::Constant *EltInit;
    if (ElementNo < ILE->getNumInits())
      EltInit = CGM.EmitConstantExpr(ILE->getInit(ElementNo++),
                                     Field->getType(), CGF);
    else
      EltInit = CGM.EmitNullConstant(Field->getType());

    if (!EltInit)
      return false;
    
    if (!Field->isBitField()) {
      // Handle non-bitfield members.
      AppendField(*Field, Layout.getFieldOffset(FieldNo), EltInit);
    } else {
      // Otherwise we have a bitfield.
      AppendBitField(*Field, Layout.getFieldOffset(FieldNo),
                     cast<llvm::ConstantInt>(EltInit));
    }
  }

  return true;
}

namespace {
struct BaseInfo {
  BaseInfo(const CXXRecordDecl *Decl, CharUnits Offset, unsigned Index)
    : Decl(Decl), Offset(Offset), Index(Index) {
  }

  const CXXRecordDecl *Decl;
  CharUnits Offset;
  unsigned Index;

  bool operator<(const BaseInfo &O) const { return Offset < O.Offset; }
};
}

void ConstStructBuilder::Build(const APValue &Val, const RecordDecl *RD,
                               bool IsPrimaryBase,
                               const CXXRecordDecl *VTableClass,
                               CharUnits Offset) {
  const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

  if (const CXXRecordDecl *CD = dyn_cast<CXXRecordDecl>(RD)) {
    // Add a vtable pointer, if we need one and it hasn't already been added.
    if (CD->isDynamicClass() && !IsPrimaryBase) {
      llvm::Constant *VTableAddressPoint =
          CGM.getCXXABI().getVTableAddressPointForConstExpr(
              BaseSubobject(CD, Offset), VTableClass);
      AppendBytes(Offset, VTableAddressPoint);
    }

    // Accumulate and sort bases, in order to visit them in address order, which
    // may not be the same as declaration order.
    SmallVector<BaseInfo, 8> Bases;
    Bases.reserve(CD->getNumBases());
    unsigned BaseNo = 0;
    for (CXXRecordDecl::base_class_const_iterator Base = CD->bases_begin(),
         BaseEnd = CD->bases_end(); Base != BaseEnd; ++Base, ++BaseNo) {
      assert(!Base->isVirtual() && "should not have virtual bases here");
      const CXXRecordDecl *BD = Base->getType()->getAsCXXRecordDecl();
      CharUnits BaseOffset = Layout.getBaseClassOffset(BD);
      Bases.push_back(BaseInfo(BD, BaseOffset, BaseNo));
    }
    std::stable_sort(Bases.begin(), Bases.end());

    for (unsigned I = 0, N = Bases.size(); I != N; ++I) {
      BaseInfo &Base = Bases[I];

      bool IsPrimaryBase = Layout.getPrimaryBase() == Base.Decl;
      Build(Val.getStructBase(Base.Index), Base.Decl, IsPrimaryBase,
            VTableClass, Offset + Base.Offset);
    }
  }

  unsigned FieldNo = 0;
  uint64_t OffsetBits = CGM.getContext().toBits(Offset);

  for (RecordDecl::field_iterator Field = RD->field_begin(),
       FieldEnd = RD->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    // If this is a union, skip all the fields that aren't being initialized.
    if (RD->isUnion() && Val.getUnionField() != *Field)
      continue;

    // Don't emit anonymous bitfields, they just affect layout.
    if (Field->isUnnamedBitfield())
      continue;

    // Emit the value of the initializer.
    const APValue &FieldValue =
      RD->isUnion() ? Val.getUnionValue() : Val.getStructField(FieldNo);
    llvm::Constant *EltInit =
      CGM.EmitConstantValueForMemory(FieldValue, Field->getType(), CGF);
    assert(EltInit && "EmitConstantValue can't fail");

    if (!Field->isBitField()) {
      // Handle non-bitfield members.
      AppendField(*Field, Layout.getFieldOffset(FieldNo) + OffsetBits, EltInit);
    } else {
      // Otherwise we have a bitfield.
      AppendBitField(*Field, Layout.getFieldOffset(FieldNo) + OffsetBits,
                     cast<llvm::ConstantInt>(EltInit));
    }
  }
}

llvm::Constant *ConstStructBuilder::Finalize(QualType Ty) {
  RecordDecl *RD = Ty->getAs<RecordType>()->getDecl();
  const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

  CharUnits LayoutSizeInChars = Layout.getSize();

  if (NextFieldOffsetInChars > LayoutSizeInChars) {
    // If the struct is bigger than the size of the record type,
    // we must have a flexible array member at the end.
    assert(RD->hasFlexibleArrayMember() &&
           "Must have flexible array member if struct is bigger than type!");

    // No tail padding is necessary.
  } else {
    // Append tail padding if necessary.
    AppendTailPadding(LayoutSizeInChars);

    CharUnits LLVMSizeInChars =
      NextFieldOffsetInChars.RoundUpToAlignment(LLVMStructAlignment);

    // Check if we need to convert the struct to a packed struct.
    if (NextFieldOffsetInChars <= LayoutSizeInChars &&
        LLVMSizeInChars > LayoutSizeInChars) {
      assert(!Packed && "Size mismatch!");

      ConvertStructToPacked();
      assert(NextFieldOffsetInChars <= LayoutSizeInChars &&
             "Converting to packed did not help!");
    }

    assert(LayoutSizeInChars == NextFieldOffsetInChars &&
           "Tail padding mismatch!");
  }

  // Pick the type to use.  If the type is layout identical to the ConvertType
  // type then use it, otherwise use whatever the builder produced for us.
  llvm::StructType *STy =
      llvm::ConstantStruct::getTypeForElements(CGM.getLLVMContext(),
                                               Elements, Packed);
  llvm::Type *ValTy = CGM.getTypes().ConvertType(Ty);
  if (llvm::StructType *ValSTy = dyn_cast<llvm::StructType>(ValTy)) {
    if (ValSTy->isLayoutIdentical(STy))
      STy = ValSTy;
  }

  llvm::Constant *Result = llvm::ConstantStruct::get(STy, Elements);

  assert(NextFieldOffsetInChars.RoundUpToAlignment(getAlignment(Result)) ==
         getSizeInChars(Result) && "Size mismatch!");

  return Result;
}

llvm::Constant *ConstStructBuilder::BuildStruct(CodeGenModule &CGM,
                                                CodeGenFunction *CGF,
                                                InitListExpr *ILE) {
  ConstStructBuilder Builder(CGM, CGF);

  if (!Builder.Build(ILE))
    return 0;

  return Builder.Finalize(ILE->getType());
}

llvm::Constant *ConstStructBuilder::BuildStruct(CodeGenModule &CGM,
                                                CodeGenFunction *CGF,
                                                const APValue &Val,
                                                QualType ValTy) {
  ConstStructBuilder Builder(CGM, CGF);

  const RecordDecl *RD = ValTy->castAs<RecordType>()->getDecl();
  const CXXRecordDecl *CD = dyn_cast<CXXRecordDecl>(RD);
  Builder.Build(Val, RD, false, CD, CharUnits::Zero());

  return Builder.Finalize(ValTy);
}


//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//

/// This class only needs to handle two cases:
/// 1) Literals (this is used by APValue emission to emit literals).
/// 2) Arrays, structs and unions (outside C++11 mode, we don't currently
///    constant fold these types).
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

  llvm::Constant *
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *PE) {
    return Visit(PE->getReplacement());
  }

  llvm::Constant *VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    return Visit(GE->getResultExpr());
  }

  llvm::Constant *VisitChooseExpr(ChooseExpr *CE) {
    return Visit(CE->getChosenSubExpr());
  }

  llvm::Constant *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return Visit(E->getInitializer());
  }

  llvm::Constant *VisitCastExpr(CastExpr* E) {
    Expr *subExpr = E->getSubExpr();
    llvm::Constant *C = CGM.EmitConstantExpr(subExpr, subExpr->getType(), CGF);
    if (!C) return 0;

    llvm::Type *destType = ConvertType(E->getType());

    switch (E->getCastKind()) {
    case CK_ToUnion: {
      // GCC cast to union extension
      assert(E->getType()->isUnionType() &&
             "Destination type is not union type!");

      // Build a struct with the union sub-element as the first member,
      // and padded to the appropriate size
      SmallVector<llvm::Constant*, 2> Elts;
      SmallVector<llvm::Type*, 2> Types;
      Elts.push_back(C);
      Types.push_back(C->getType());
      unsigned CurSize = CGM.getDataLayout().getTypeAllocSize(C->getType());
      unsigned TotalSize = CGM.getDataLayout().getTypeAllocSize(destType);

      assert(CurSize <= TotalSize && "Union size mismatch!");
      if (unsigned NumPadBytes = TotalSize - CurSize) {
        llvm::Type *Ty = CGM.Int8Ty;
        if (NumPadBytes > 1)
          Ty = llvm::ArrayType::get(Ty, NumPadBytes);

        Elts.push_back(llvm::UndefValue::get(Ty));
        Types.push_back(Ty);
      }

      llvm::StructType* STy =
        llvm::StructType::get(C->getType()->getContext(), Types, false);
      return llvm::ConstantStruct::get(STy, Elts);
    }

    case CK_AddressSpaceConversion:
      return llvm::ConstantExpr::getAddrSpaceCast(C, destType);

    case CK_LValueToRValue:
    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_ConstructorConversion:
      return C;

    case CK_Dependent: llvm_unreachable("saw dependent cast!");

    case CK_BuiltinFnToFnPtr:
      llvm_unreachable("builtin functions are handled elsewhere");

    case CK_ReinterpretMemberPointer:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer:
      return CGM.getCXXABI().EmitMemberPointerConversion(E, C);

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_ARCProduceObject:
    case CK_ARCConsumeObject:
    case CK_ARCReclaimReturnedObject:
    case CK_ARCExtendBlockObject:
    case CK_CopyAndAutoreleaseBlockObject:
      return 0;

    // These don't need to be handled here because Evaluate knows how to
    // evaluate them in the cases where they can be folded.
    case CK_BitCast:
    case CK_ToVoid:
    case CK_Dynamic:
    case CK_LValueBitCast:
    case CK_NullToMemberPointer:
    case CK_UserDefinedConversion:
    case CK_CPointerToObjCPointerCast:
    case CK_BlockPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_MemberPointerToBoolean:
    case CK_VectorSplat:
    case CK_FloatingRealToComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingComplexToBoolean:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToIntegralComplex:
    case CK_IntegralRealToComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralComplexToBoolean:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToFloatingComplex:
    case CK_PointerToIntegral:
    case CK_PointerToBoolean:
    case CK_NullToPointer:
    case CK_IntegralCast:
    case CK_IntegralToPointer:
    case CK_IntegralToBoolean:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingToBoolean:
    case CK_FloatingCast:
    case CK_ZeroToOCLEvent:
      return 0;
    }
    llvm_unreachable("Invalid CastKind");
  }

  llvm::Constant *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }

  llvm::Constant *VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    // No need for a DefaultInitExprScope: we don't handle 'this' in a
    // constant expression.
    return Visit(DIE->getExpr());
  }

  llvm::Constant *VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
    return Visit(E->GetTemporaryExpr());
  }

  llvm::Constant *EmitArrayInitialization(InitListExpr *ILE) {
    if (ILE->isStringLiteralInit())
      return Visit(ILE->getInit(0));

    llvm::ArrayType *AType =
        cast<llvm::ArrayType>(ConvertType(ILE->getType()));
    llvm::Type *ElemTy = AType->getElementType();
    unsigned NumInitElements = ILE->getNumInits();
    unsigned NumElements = AType->getNumElements();

    // Initialising an array requires us to automatically
    // initialise any elements that have not been initialised explicitly
    unsigned NumInitableElts = std::min(NumInitElements, NumElements);

    // Copy initializer elements.
    std::vector<llvm::Constant*> Elts;
    Elts.reserve(NumInitableElts + NumElements);

    bool RewriteType = false;
    for (unsigned i = 0; i < NumInitableElts; ++i) {
      Expr *Init = ILE->getInit(i);
      llvm::Constant *C = CGM.EmitConstantExpr(Init, Init->getType(), CGF);
      if (!C)
        return 0;
      RewriteType |= (C->getType() != ElemTy);
      Elts.push_back(C);
    }

    // Initialize remaining array elements.
    // FIXME: This doesn't handle member pointers correctly!
    llvm::Constant *fillC;
    if (Expr *filler = ILE->getArrayFiller())
      fillC = CGM.EmitConstantExpr(filler, filler->getType(), CGF);
    else
      fillC = llvm::Constant::getNullValue(ElemTy);
    if (!fillC)
      return 0;
    RewriteType |= (fillC->getType() != ElemTy);
    Elts.resize(NumElements, fillC);

    if (RewriteType) {
      // FIXME: Try to avoid packing the array
      std::vector<llvm::Type*> Types;
      Types.reserve(NumInitableElts + NumElements);
      for (unsigned i = 0, e = Elts.size(); i < e; ++i)
        Types.push_back(Elts[i]->getType());
      llvm::StructType *SType = llvm::StructType::get(AType->getContext(),
                                                            Types, true);
      return llvm::ConstantStruct::get(SType, Elts);
    }

    return llvm::ConstantArray::get(AType, Elts);
  }

  llvm::Constant *EmitRecordInitialization(InitListExpr *ILE) {
    return ConstStructBuilder::BuildStruct(CGM, CGF, ILE);
  }

  llvm::Constant *VisitImplicitValueInitExpr(ImplicitValueInitExpr* E) {
    return CGM.EmitNullConstant(E->getType());
  }

  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE);

    if (ILE->getType()->isRecordType())
      return EmitRecordInitialization(ILE);

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
      assert(E->getConstructor()->isCopyOrMoveConstructor() &&
             "trivial ctor has argument but isn't a copy/move ctor");

      Expr *Arg = E->getArg(0);
      assert(CGM.getContext().hasSameUnqualifiedType(Ty, Arg->getType()) &&
             "argument to copy ctor is of wrong type");

      return Visit(Arg);
    }

    return CGM.EmitNullConstant(Ty);
  }

  llvm::Constant *VisitStringLiteral(StringLiteral *E) {
    return CGM.GetConstantArrayFromStringLiteral(E);
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
    return llvm::ConstantDataArray::getString(VMContext, Str, false);
  }

  llvm::Constant *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }

  // Utility methods
  llvm::Type *ConvertType(QualType T) {
    return CGM.getTypes().ConvertType(T);
  }

public:
  llvm::Constant *EmitLValue(APValue::LValueBase LVBase) {
    if (const ValueDecl *Decl = LVBase.dyn_cast<const ValueDecl*>()) {
      if (Decl->hasAttr<WeakRefAttr>())
        return CGM.GetWeakRefReference(Decl);
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Decl))
        return CGM.GetAddrOfFunction(FD);
      if (const VarDecl* VD = dyn_cast<VarDecl>(Decl)) {
        // We can never refer to a variable with local storage.
        if (!VD->hasLocalStorage()) {
          if (VD->isFileVarDecl() || VD->hasExternalStorage())
            return CGM.GetAddrOfGlobalVar(VD);
          else if (VD->isLocalVarDecl())
            return CGM.getStaticLocalDeclAddress(VD);
        }
      }
      return 0;
    }

    Expr *E = const_cast<Expr*>(LVBase.get<const Expr*>());
    switch (E->getStmtClass()) {
    default: break;
    case Expr::CompoundLiteralExprClass: {
      // Note that due to the nature of compound literals, this is guaranteed
      // to be the only use of the variable, so we just generate it here.
      CompoundLiteralExpr *CLE = cast<CompoundLiteralExpr>(E);
      llvm::Constant* C = CGM.EmitConstantExpr(CLE->getInitializer(),
                                               CLE->getType(), CGF);
      // FIXME: "Leaked" on failure.
      if (C)
        C = new llvm::GlobalVariable(CGM.getModule(), C->getType(),
                                     E->getType().isConstant(CGM.getContext()),
                                     llvm::GlobalValue::InternalLinkage,
                                     C, ".compoundliteral", 0,
                                     llvm::GlobalVariable::NotThreadLocal,
                          CGM.getContext().getTargetAddressSpace(E->getType()));
      return C;
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
        LValue Res = CGF->EmitPredefinedLValue(cast<PredefinedExpr>(E));
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
      unsigned builtin = CE->isBuiltinCall();
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
    case Expr::CXXTypeidExprClass: {
      CXXTypeidExpr *Typeid = cast<CXXTypeidExpr>(E);
      QualType T;
      if (Typeid->isTypeOperand())
        T = Typeid->getTypeOperand(CGM.getContext());
      else
        T = Typeid->getExprOperand()->getType();
      return CGM.GetAddrOfRTTIDescriptor(T);
    }
    case Expr::CXXUuidofExprClass: {
      return CGM.GetAddrOfUuidDescriptor(cast<CXXUuidofExpr>(E));
    }
    case Expr::MaterializeTemporaryExprClass: {
      MaterializeTemporaryExpr *MTE = cast<MaterializeTemporaryExpr>(E);
      assert(MTE->getStorageDuration() == SD_Static);
      SmallVector<const Expr *, 2> CommaLHSs;
      SmallVector<SubobjectAdjustment, 2> Adjustments;
      const Expr *Inner = MTE->GetTemporaryExpr()
          ->skipRValueSubobjectAdjustments(CommaLHSs, Adjustments);
      return CGM.GetAddrOfGlobalTemporary(MTE, Inner);
    }
    }

    return 0;
  }
};

}  // end anonymous namespace.

llvm::Constant *CodeGenModule::EmitConstantInit(const VarDecl &D,
                                                CodeGenFunction *CGF) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!D.hasLocalStorage()) {
    QualType Ty = D.getType();
    if (Ty->isArrayType())
      Ty = Context.getBaseElementType(Ty);
    if (Ty->isRecordType())
      if (const CXXConstructExpr *E =
          dyn_cast_or_null<CXXConstructExpr>(D.getInit())) {
        const CXXConstructorDecl *CD = E->getConstructor();
        if (CD->isTrivial() && CD->isDefaultConstructor())
          return EmitNullConstant(D.getType());
      }
  }
  
  if (const APValue *Value = D.evaluateValue())
    return EmitConstantValueForMemory(*Value, D.getType(), CGF);

  // FIXME: Implement C++11 [basic.start.init]p2: if the initializer of a
  // reference is a constant expression, and the reference binds to a temporary,
  // then constant initialization is performed. ConstExprEmitter will
  // incorrectly emit a prvalue constant in this case, and the calling code
  // interprets that as the (pointer) value of the reference, rather than the
  // desired value of the referee.
  if (D.getType()->isReferenceType())
    return 0;

  const Expr *E = D.getInit();
  assert(E && "No initializer to emit");

  llvm::Constant* C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
  if (C && C->getType()->isIntegerTy(1)) {
    llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}

llvm::Constant *CodeGenModule::EmitConstantExpr(const Expr *E,
                                                QualType DestType,
                                                CodeGenFunction *CGF) {
  Expr::EvalResult Result;

  bool Success = false;

  if (DestType->isReferenceType())
    Success = E->EvaluateAsLValue(Result, Context);
  else
    Success = E->EvaluateAsRValue(Result, Context);

  llvm::Constant *C = 0;
  if (Success && !Result.HasSideEffects)
    C = EmitConstantValue(Result.Val, DestType, CGF);
  else
    C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));

  if (C && C->getType()->isIntegerTy(1)) {
    llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}

llvm::Constant *CodeGenModule::EmitConstantValue(const APValue &Value,
                                                 QualType DestType,
                                                 CodeGenFunction *CGF) {
  switch (Value.getKind()) {
  case APValue::Uninitialized:
    llvm_unreachable("Constant expressions should be initialized.");
  case APValue::LValue: {
    llvm::Type *DestTy = getTypes().ConvertTypeForMem(DestType);
    llvm::Constant *Offset =
      llvm::ConstantInt::get(Int64Ty, Value.getLValueOffset().getQuantity());

    llvm::Constant *C;
    if (APValue::LValueBase LVBase = Value.getLValueBase()) {
      // An array can be represented as an lvalue referring to the base.
      if (isa<llvm::ArrayType>(DestTy)) {
        assert(Offset->isNullValue() && "offset on array initializer");
        return ConstExprEmitter(*this, CGF).Visit(
          const_cast<Expr*>(LVBase.get<const Expr*>()));
      }

      C = ConstExprEmitter(*this, CGF).EmitLValue(LVBase);

      // Apply offset if necessary.
      if (!Offset->isNullValue()) {
        llvm::Constant *Casted = llvm::ConstantExpr::getBitCast(C, Int8PtrTy);
        Casted = llvm::ConstantExpr::getGetElementPtr(Casted, Offset);
        C = llvm::ConstantExpr::getPointerCast(Casted, C->getType());
      }

      // Convert to the appropriate type; this could be an lvalue for
      // an integer.
      if (isa<llvm::PointerType>(DestTy))
        return llvm::ConstantExpr::getPointerCast(C, DestTy);

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
  case APValue::Int:
    return llvm::ConstantInt::get(VMContext, Value.getInt());
  case APValue::ComplexInt: {
    llvm::Constant *Complex[2];

    Complex[0] = llvm::ConstantInt::get(VMContext,
                                        Value.getComplexIntReal());
    Complex[1] = llvm::ConstantInt::get(VMContext,
                                        Value.getComplexIntImag());

    // FIXME: the target may want to specify that this is packed.
    llvm::StructType *STy = llvm::StructType::get(Complex[0]->getType(),
                                                  Complex[1]->getType(),
                                                  NULL);
    return llvm::ConstantStruct::get(STy, Complex);
  }
  case APValue::Float: {
    const llvm::APFloat &Init = Value.getFloat();
    if (&Init.getSemantics() == &llvm::APFloat::IEEEhalf &&
         !Context.getLangOpts().NativeHalfType)
      return llvm::ConstantInt::get(VMContext, Init.bitcastToAPInt());
    else
      return llvm::ConstantFP::get(VMContext, Init);
  }
  case APValue::ComplexFloat: {
    llvm::Constant *Complex[2];

    Complex[0] = llvm::ConstantFP::get(VMContext,
                                       Value.getComplexFloatReal());
    Complex[1] = llvm::ConstantFP::get(VMContext,
                                       Value.getComplexFloatImag());

    // FIXME: the target may want to specify that this is packed.
    llvm::StructType *STy = llvm::StructType::get(Complex[0]->getType(),
                                                  Complex[1]->getType(),
                                                  NULL);
    return llvm::ConstantStruct::get(STy, Complex);
  }
  case APValue::Vector: {
    SmallVector<llvm::Constant *, 4> Inits;
    unsigned NumElts = Value.getVectorLength();

    for (unsigned i = 0; i != NumElts; ++i) {
      const APValue &Elt = Value.getVectorElt(i);
      if (Elt.isInt())
        Inits.push_back(llvm::ConstantInt::get(VMContext, Elt.getInt()));
      else
        Inits.push_back(llvm::ConstantFP::get(VMContext, Elt.getFloat()));
    }
    return llvm::ConstantVector::get(Inits);
  }
  case APValue::AddrLabelDiff: {
    const AddrLabelExpr *LHSExpr = Value.getAddrLabelDiffLHS();
    const AddrLabelExpr *RHSExpr = Value.getAddrLabelDiffRHS();
    llvm::Constant *LHS = EmitConstantExpr(LHSExpr, LHSExpr->getType(), CGF);
    llvm::Constant *RHS = EmitConstantExpr(RHSExpr, RHSExpr->getType(), CGF);

    // Compute difference
    llvm::Type *ResultType = getTypes().ConvertType(DestType);
    LHS = llvm::ConstantExpr::getPtrToInt(LHS, IntPtrTy);
    RHS = llvm::ConstantExpr::getPtrToInt(RHS, IntPtrTy);
    llvm::Constant *AddrLabelDiff = llvm::ConstantExpr::getSub(LHS, RHS);

    // LLVM is a bit sensitive about the exact format of the
    // address-of-label difference; make sure to truncate after
    // the subtraction.
    return llvm::ConstantExpr::getTruncOrBitCast(AddrLabelDiff, ResultType);
  }
  case APValue::Struct:
  case APValue::Union:
    return ConstStructBuilder::BuildStruct(*this, CGF, Value, DestType);
  case APValue::Array: {
    const ArrayType *CAT = Context.getAsArrayType(DestType);
    unsigned NumElements = Value.getArraySize();
    unsigned NumInitElts = Value.getArrayInitializedElts();

    std::vector<llvm::Constant*> Elts;
    Elts.reserve(NumElements);

    // Emit array filler, if there is one.
    llvm::Constant *Filler = 0;
    if (Value.hasArrayFiller())
      Filler = EmitConstantValueForMemory(Value.getArrayFiller(),
                                          CAT->getElementType(), CGF);

    // Emit initializer elements.
    llvm::Type *CommonElementType = 0;
    for (unsigned I = 0; I < NumElements; ++I) {
      llvm::Constant *C = Filler;
      if (I < NumInitElts)
        C = EmitConstantValueForMemory(Value.getArrayInitializedElt(I),
                                       CAT->getElementType(), CGF);
      else
        assert(Filler && "Missing filler for implicit elements of initializer");
      if (I == 0)
        CommonElementType = C->getType();
      else if (C->getType() != CommonElementType)
        CommonElementType = 0;
      Elts.push_back(C);
    }

    if (!CommonElementType) {
      // FIXME: Try to avoid packing the array
      std::vector<llvm::Type*> Types;
      Types.reserve(NumElements);
      for (unsigned i = 0, e = Elts.size(); i < e; ++i)
        Types.push_back(Elts[i]->getType());
      llvm::StructType *SType = llvm::StructType::get(VMContext, Types, true);
      return llvm::ConstantStruct::get(SType, Elts);
    }

    llvm::ArrayType *AType =
      llvm::ArrayType::get(CommonElementType, NumElements);
    return llvm::ConstantArray::get(AType, Elts);
  }
  case APValue::MemberPointer:
    return getCXXABI().EmitMemberPointer(Value, DestType);
  }
  llvm_unreachable("Unknown APValue kind");
}

llvm::Constant *
CodeGenModule::EmitConstantValueForMemory(const APValue &Value,
                                          QualType DestType,
                                          CodeGenFunction *CGF) {
  llvm::Constant *C = EmitConstantValue(Value, DestType, CGF);
  if (C->getType()->isIntegerTy(1)) {
    llvm::Type *BoolTy = getTypes().ConvertTypeForMem(DestType);
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}

llvm::Constant *
CodeGenModule::GetAddrOfConstantCompoundLiteral(const CompoundLiteralExpr *E) {
  assert(E->isFileScope() && "not a file-scope compound literal expr");
  return ConstExprEmitter(*this, 0).EmitLValue(E);
}

llvm::Constant *
CodeGenModule::getMemberPointerConstant(const UnaryOperator *uo) {
  // Member pointer constants always have a very particular form.
  const MemberPointerType *type = cast<MemberPointerType>(uo->getType());
  const ValueDecl *decl = cast<DeclRefExpr>(uo->getSubExpr())->getDecl();

  // A member function pointer.
  if (const CXXMethodDecl *method = dyn_cast<CXXMethodDecl>(decl))
    return getCXXABI().EmitMemberPointer(method);

  // Otherwise, a member data pointer.
  uint64_t fieldOffset = getContext().getFieldOffset(decl);
  CharUnits chars = getContext().toCharUnitsFromBits((int64_t) fieldOffset);
  return getCXXABI().EmitMemberDataPointer(type, chars);
}

static void
FillInNullDataMemberPointers(CodeGenModule &CGM, QualType T,
                             SmallVectorImpl<llvm::Constant *> &Elements,
                             uint64_t StartOffset) {
  assert(StartOffset % CGM.getContext().getCharWidth() == 0 && 
         "StartOffset not byte aligned!");

  if (CGM.getTypes().isZeroInitializable(T))
    return;

  if (const ConstantArrayType *CAT = 
        CGM.getContext().getAsConstantArrayType(T)) {
    QualType ElementTy = CAT->getElementType();
    uint64_t ElementSize = CGM.getContext().getTypeSize(ElementTy);
    
    for (uint64_t I = 0, E = CAT->getSize().getZExtValue(); I != E; ++I) {
      FillInNullDataMemberPointers(CGM, ElementTy, Elements,
                                   StartOffset + I * ElementSize);
    }
  } else if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

    // Go through all bases and fill in any null pointer to data members.
    for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
      if (I->isVirtual()) {
        // Ignore virtual bases.
        continue;
      }
      
      const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      // Ignore empty bases.
      if (BaseDecl->isEmpty())
        continue;
      
      // Ignore bases that don't have any pointer to data members.
      if (CGM.getTypes().isZeroInitializable(BaseDecl))
        continue;

      uint64_t BaseOffset =
        CGM.getContext().toBits(Layout.getBaseClassOffset(BaseDecl));
      FillInNullDataMemberPointers(CGM, I->getType(),
                                   Elements, StartOffset + BaseOffset);
    }
    
    // Visit all fields.
    unsigned FieldNo = 0;
    for (RecordDecl::field_iterator I = RD->field_begin(),
         E = RD->field_end(); I != E; ++I, ++FieldNo) {
      QualType FieldType = I->getType();
      
      if (CGM.getTypes().isZeroInitializable(FieldType))
        continue;

      uint64_t FieldOffset = StartOffset + Layout.getFieldOffset(FieldNo);
      FillInNullDataMemberPointers(CGM, FieldType, Elements, FieldOffset);
    }
  } else {
    assert(T->isMemberPointerType() && "Should only see member pointers here!");
    assert(!T->getAs<MemberPointerType>()->getPointeeType()->isFunctionType() &&
           "Should only see pointers to data members here!");
  
    CharUnits StartIndex = CGM.getContext().toCharUnitsFromBits(StartOffset);
    CharUnits EndIndex = StartIndex + CGM.getContext().getTypeSizeInChars(T);

    // FIXME: hardcodes Itanium member pointer representation!
    llvm::Constant *NegativeOne =
      llvm::ConstantInt::get(CGM.Int8Ty, -1ULL, /*isSigned*/true);

    // Fill in the null data member pointer.
    for (CharUnits I = StartIndex; I != EndIndex; ++I)
      Elements[I.getQuantity()] = NegativeOne;
  }
}

static llvm::Constant *EmitNullConstantForBase(CodeGenModule &CGM,
                                               llvm::Type *baseType,
                                               const CXXRecordDecl *base);

static llvm::Constant *EmitNullConstant(CodeGenModule &CGM,
                                        const CXXRecordDecl *record,
                                        bool asCompleteObject) {
  const CGRecordLayout &layout = CGM.getTypes().getCGRecordLayout(record);
  llvm::StructType *structure =
    (asCompleteObject ? layout.getLLVMType()
                      : layout.getBaseSubobjectLLVMType());

  unsigned numElements = structure->getNumElements();
  std::vector<llvm::Constant *> elements(numElements);

  // Fill in all the bases.
  for (CXXRecordDecl::base_class_const_iterator
         I = record->bases_begin(), E = record->bases_end(); I != E; ++I) {
    if (I->isVirtual()) {
      // Ignore virtual bases; if we're laying out for a complete
      // object, we'll lay these out later.
      continue;
    }

    const CXXRecordDecl *base = 
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

    // Ignore empty bases.
    if (base->isEmpty())
      continue;
    
    unsigned fieldIndex = layout.getNonVirtualBaseLLVMFieldNo(base);
    llvm::Type *baseType = structure->getElementType(fieldIndex);
    elements[fieldIndex] = EmitNullConstantForBase(CGM, baseType, base);
  }

  // Fill in all the fields.
  for (RecordDecl::field_iterator I = record->field_begin(),
         E = record->field_end(); I != E; ++I) {
    const FieldDecl *field = *I;

    // Fill in non-bitfields. (Bitfields always use a zero pattern, which we
    // will fill in later.)
    if (!field->isBitField()) {
      unsigned fieldIndex = layout.getLLVMFieldNo(field);
      elements[fieldIndex] = CGM.EmitNullConstant(field->getType());
    }

    // For unions, stop after the first named field.
    if (record->isUnion() && field->getDeclName())
      break;
  }

  // Fill in the virtual bases, if we're working with the complete object.
  if (asCompleteObject) {
    for (CXXRecordDecl::base_class_const_iterator
           I = record->vbases_begin(), E = record->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *base = 
        cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

      // Ignore empty bases.
      if (base->isEmpty())
        continue;

      unsigned fieldIndex = layout.getVirtualBaseIndex(base);

      // We might have already laid this field out.
      if (elements[fieldIndex]) continue;

      llvm::Type *baseType = structure->getElementType(fieldIndex);
      elements[fieldIndex] = EmitNullConstantForBase(CGM, baseType, base);
    }
  }

  // Now go through all other fields and zero them out.
  for (unsigned i = 0; i != numElements; ++i) {
    if (!elements[i])
      elements[i] = llvm::Constant::getNullValue(structure->getElementType(i));
  }
  
  return llvm::ConstantStruct::get(structure, elements);
}

/// Emit the null constant for a base subobject.
static llvm::Constant *EmitNullConstantForBase(CodeGenModule &CGM,
                                               llvm::Type *baseType,
                                               const CXXRecordDecl *base) {
  const CGRecordLayout &baseLayout = CGM.getTypes().getCGRecordLayout(base);

  // Just zero out bases that don't have any pointer to data members.
  if (baseLayout.isZeroInitializableAsBase())
    return llvm::Constant::getNullValue(baseType);

  // If the base type is a struct, we can just use its null constant.
  if (isa<llvm::StructType>(baseType)) {
    return EmitNullConstant(CGM, base, /*complete*/ false);
  }

  // Otherwise, some bases are represented as arrays of i8 if the size
  // of the base is smaller than its corresponding LLVM type.  Figure
  // out how many elements this base array has.
  llvm::ArrayType *baseArrayType = cast<llvm::ArrayType>(baseType);
  unsigned numBaseElements = baseArrayType->getNumElements();

  // Fill in null data member pointers.
  SmallVector<llvm::Constant *, 16> baseElements(numBaseElements);
  FillInNullDataMemberPointers(CGM, CGM.getContext().getTypeDeclType(base),
                               baseElements, 0);

  // Now go through all other elements and zero them out.
  if (numBaseElements) {
    llvm::Constant *i8_zero = llvm::Constant::getNullValue(CGM.Int8Ty);
    for (unsigned i = 0; i != numBaseElements; ++i) {
      if (!baseElements[i])
        baseElements[i] = i8_zero;
    }
  }
      
  return llvm::ConstantArray::get(baseArrayType, baseElements);
}

llvm::Constant *CodeGenModule::EmitNullConstant(QualType T) {
  if (getTypes().isZeroInitializable(T))
    return llvm::Constant::getNullValue(getTypes().ConvertTypeForMem(T));
    
  if (const ConstantArrayType *CAT = Context.getAsConstantArrayType(T)) {
    llvm::ArrayType *ATy =
      cast<llvm::ArrayType>(getTypes().ConvertTypeForMem(T));

    QualType ElementTy = CAT->getElementType();

    llvm::Constant *Element = EmitNullConstant(ElementTy);
    unsigned NumElements = CAT->getSize().getZExtValue();
    
    if (Element->isNullValue())
      return llvm::ConstantAggregateZero::get(ATy);
    
    SmallVector<llvm::Constant *, 8> Array(NumElements, Element);
    return llvm::ConstantArray::get(ATy, Array);
  }

  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    return ::EmitNullConstant(*this, RD, /*complete object*/ true);
  }

  assert(T->isMemberPointerType() && "Should only see member pointers here!");
  assert(!T->getAs<MemberPointerType>()->getPointeeType()->isFunctionType() &&
         "Should only see pointers to data members here!");
  
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  return getCXXABI().EmitNullMemberPointer(T->castAs<MemberPointerType>());
}

llvm::Constant *
CodeGenModule::EmitNullConstantForBase(const CXXRecordDecl *Record) {
  return ::EmitNullConstant(*this, Record, false);
}
