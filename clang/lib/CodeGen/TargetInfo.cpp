//===---- TargetInfo.cpp - Encapsulate target details -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CodeGenFunction.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace CodeGen;

ABIInfo::~ABIInfo() {}

void ABIArgInfo::dump() const {
  llvm::raw_ostream &OS = llvm::errs();
  OS << "(ABIArgInfo Kind=";
  switch (TheKind) {
  case Direct:
    OS << "Direct";
    break;
  case Extend:
    OS << "Extend";
    break;
  case Ignore:
    OS << "Ignore";
    break;
  case Coerce:
    OS << "Coerce Type=";
    getCoerceToType()->print(OS);
    break;
  case Indirect:
    OS << "Indirect Align=" << getIndirectAlign();
    break;
  case Expand:
    OS << "Expand";
    break;
  }
  OS << ")\n";
}

TargetCodeGenInfo::~TargetCodeGenInfo() { delete Info; }

static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays);

/// isEmptyField - Return true iff a the field is "empty", that is it
/// is an unnamed bit-field or an (array of) empty record(s).
static bool isEmptyField(ASTContext &Context, const FieldDecl *FD,
                         bool AllowArrays) {
  if (FD->isUnnamedBitfield())
    return true;

  QualType FT = FD->getType();

    // Constant arrays of empty records count as empty, strip them off.
  if (AllowArrays)
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT))
      FT = AT->getElementType();

  return isEmptyRecord(Context, FT, AllowArrays);
}

/// isEmptyRecord - Return true iff a structure contains only empty
/// fields. Note that a structure with a flexible array member is not
/// considered empty.
static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return 0;
  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return false;
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i)
    if (!isEmptyField(Context, *i, AllowArrays))
      return false;
  return true;
}

/// hasNonTrivialDestructorOrCopyConstructor - Determine if a type has either
/// a non-trivial destructor or a non-trivial copy constructor.
static bool hasNonTrivialDestructorOrCopyConstructor(const RecordType *RT) {
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return false;
  
  return !RD->hasTrivialDestructor() || !RD->hasTrivialCopyConstructor();
}

/// isRecordWithNonTrivialDestructorOrCopyConstructor - Determine if a type is
/// a record type with either a non-trivial destructor or a non-trivial copy
/// constructor.
static bool isRecordWithNonTrivialDestructorOrCopyConstructor(QualType T) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  return hasNonTrivialDestructorOrCopyConstructor(RT);
}

/// isSingleElementStruct - Determine if a structure is a "single
/// element struct", i.e. it has exactly one non-empty field or
/// exactly one field which is itself a single element
/// struct. Structures with flexible array members are never
/// considered single element structs.
///
/// \return The field declaration for the single non-empty field, if
/// it exists.
static const Type *isSingleElementStruct(QualType T, ASTContext &Context) {
  const RecordType *RT = T->getAsStructureType();
  if (!RT)
    return 0;

  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return 0;

  const Type *Found = 0;
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    const FieldDecl *FD = *i;
    QualType FT = FD->getType();

    // Ignore empty fields.
    if (isEmptyField(Context, FD, true))
      continue;

    // If we already found an element then this isn't a single-element
    // struct.
    if (Found)
      return 0;

    // Treat single element arrays as the element.
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT)) {
      if (AT->getSize().getZExtValue() != 1)
        break;
      FT = AT->getElementType();
    }

    if (!CodeGenFunction::hasAggregateLLVMType(FT)) {
      Found = FT.getTypePtr();
    } else {
      Found = isSingleElementStruct(FT, Context);
      if (!Found)
        return 0;
    }
  }

  return Found;
}

static bool is32Or64BitBasicType(QualType Ty, ASTContext &Context) {
  if (!Ty->getAs<BuiltinType>() && !Ty->isAnyPointerType() &&
      !Ty->isAnyComplexType() && !Ty->isEnumeralType() &&
      !Ty->isBlockPointerType())
    return false;

  uint64_t Size = Context.getTypeSize(Ty);
  return Size == 32 || Size == 64;
}

/// canExpandIndirectArgument - Test whether an argument type which is to be
/// passed indirectly (on the stack) would have the equivalent layout if it was
/// expanded into separate arguments. If so, we prefer to do the latter to avoid
/// inhibiting optimizations.
///
// FIXME: This predicate is missing many cases, currently it just follows
// llvm-gcc (checks that all fields are 32-bit or 64-bit primitive types). We
// should probably make this smarter, or better yet make the LLVM backend
// capable of handling it.
static bool canExpandIndirectArgument(QualType Ty, ASTContext &Context) {
  // We can only expand structure types.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT)
    return false;

  // We can only expand (C) structures.
  //
  // FIXME: This needs to be generalized to handle classes as well.
  const RecordDecl *RD = RT->getDecl();
  if (!RD->isStruct() || isa<CXXRecordDecl>(RD))
    return false;

  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    const FieldDecl *FD = *i;

    if (!is32Or64BitBasicType(FD->getType(), Context))
      return false;

    // FIXME: Reject bit-fields wholesale; there are two problems, we don't know
    // how to expand them yet, and the predicate for telling if a bitfield still
    // counts as "basic" is more complicated than what we were doing previously.
    if (FD->isBitField())
      return false;
  }

  return true;
}

static bool typeContainsSSEVector(const RecordDecl *RD, ASTContext &Context) {
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    const FieldDecl *FD = *i;

    if (FD->getType()->isVectorType() &&
        Context.getTypeSize(FD->getType()) >= 128)
      return true;

    if (const RecordType* RT = FD->getType()->getAs<RecordType>())
      if (typeContainsSSEVector(RT->getDecl(), Context))
        return true;
  }

  return false;
}

namespace {
/// DefaultABIInfo - The default implementation for ABI specific
/// details. This implementation provides information which results in
/// self-consistent and sensible LLVM IR generation, but does not
/// conform to any particular ABI.
class DefaultABIInfo : public ABIInfo {
  ABIArgInfo classifyReturnType(QualType RetTy,
                                ASTContext &Context,
                                llvm::LLVMContext &VMContext) const;

  ABIArgInfo classifyArgumentType(QualType RetTy,
                                  ASTContext &Context,
                                  llvm::LLVMContext &VMContext) const;

  virtual void computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                           llvm::LLVMContext &VMContext) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), Context,
                                            VMContext);
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type, Context, VMContext);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class DefaultTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  DefaultTargetCodeGenInfo():TargetCodeGenInfo(new DefaultABIInfo()) {}
};

llvm::Value *DefaultABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  return 0;
}

ABIArgInfo DefaultABIInfo::classifyArgumentType(QualType Ty,
                                                ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  if (CodeGenFunction::hasAggregateLLVMType(Ty))
    return ABIArgInfo::getIndirect(0);
  
  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  return (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

/// X86_32ABIInfo - The X86-32 ABI information.
class X86_32ABIInfo : public ABIInfo {
  ASTContext &Context;
  bool IsDarwinVectorABI;
  bool IsSmallStructInRegABI;

  static bool isRegisterSize(unsigned Size) {
    return (Size == 8 || Size == 16 || Size == 32 || Size == 64);
  }

  static bool shouldReturnTypeInRegister(QualType Ty, ASTContext &Context);

  static unsigned getIndirectArgumentAlignment(QualType Ty,
                                               ASTContext &Context);

public:
  ABIArgInfo classifyReturnType(QualType RetTy,
                                ASTContext &Context,
                                llvm::LLVMContext &VMContext) const;

  ABIArgInfo classifyArgumentType(QualType RetTy,
                                  ASTContext &Context,
                                  llvm::LLVMContext &VMContext) const;

  virtual void computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                           llvm::LLVMContext &VMContext) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), Context,
                                            VMContext);
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type, Context, VMContext);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;

  X86_32ABIInfo(ASTContext &Context, bool d, bool p)
    : ABIInfo(), Context(Context), IsDarwinVectorABI(d),
      IsSmallStructInRegABI(p) {}
};

class X86_32TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_32TargetCodeGenInfo(ASTContext &Context, bool d, bool p)
    :TargetCodeGenInfo(new X86_32ABIInfo(Context, d, p)) {}

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const;

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    // Darwin uses different dwarf register numbers for EH.
    if (CGM.isTargetDarwin()) return 5;

    return 4;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;
};

}

/// shouldReturnTypeInRegister - Determine if the given type should be
/// passed in a register (for the Darwin ABI).
bool X86_32ABIInfo::shouldReturnTypeInRegister(QualType Ty,
                                               ASTContext &Context) {
  uint64_t Size = Context.getTypeSize(Ty);

  // Type must be register sized.
  if (!isRegisterSize(Size))
    return false;

  if (Ty->isVectorType()) {
    // 64- and 128- bit vectors inside structures are not returned in
    // registers.
    if (Size == 64 || Size == 128)
      return false;

    return true;
  }

  // If this is a builtin, pointer, enum, or complex type, it is ok.
  if (Ty->getAs<BuiltinType>() || Ty->isAnyPointerType() || 
      Ty->isAnyComplexType() || Ty->isEnumeralType() ||
      Ty->isBlockPointerType())
    return true;

  // Arrays are treated like records.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty))
    return shouldReturnTypeInRegister(AT->getElementType(), Context);

  // Otherwise, it must be a record type.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT) return false;

  // FIXME: Traverse bases here too.

  // Structure types are passed in register if all fields would be
  // passed in a register.
  for (RecordDecl::field_iterator i = RT->getDecl()->field_begin(),
         e = RT->getDecl()->field_end(); i != e; ++i) {
    const FieldDecl *FD = *i;

    // Empty fields are ignored.
    if (isEmptyField(Context, FD, true))
      continue;

    // Check fields recursively.
    if (!shouldReturnTypeInRegister(FD->getType(), Context))
      return false;
  }

  return true;
}

ABIArgInfo X86_32ABIInfo::classifyReturnType(QualType RetTy,
                                            ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  if (RetTy->isVoidType()) {
    return ABIArgInfo::getIgnore();
  } else if (const VectorType *VT = RetTy->getAs<VectorType>()) {
    // On Darwin, some vectors are returned in registers.
    if (IsDarwinVectorABI) {
      uint64_t Size = Context.getTypeSize(RetTy);

      // 128-bit vectors are a special case; they are returned in
      // registers and we need to make sure to pick a type the LLVM
      // backend will like.
      if (Size == 128)
        return ABIArgInfo::getCoerce(llvm::VectorType::get(
                  llvm::Type::getInt64Ty(VMContext), 2));

      // Always return in register if it fits in a general purpose
      // register, or if it is 64 bits and has a single element.
      if ((Size == 8 || Size == 16 || Size == 32) ||
          (Size == 64 && VT->getNumElements() == 1))
        return ABIArgInfo::getCoerce(llvm::IntegerType::get(VMContext, Size));

      return ABIArgInfo::getIndirect(0);
    }

    return ABIArgInfo::getDirect();
  } else if (CodeGenFunction::hasAggregateLLVMType(RetTy)) {
    if (const RecordType *RT = RetTy->getAs<RecordType>()) {
      // Structures with either a non-trivial destructor or a non-trivial
      // copy constructor are always indirect.
      if (hasNonTrivialDestructorOrCopyConstructor(RT))
        return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
      
      // Structures with flexible arrays are always indirect.
      if (RT->getDecl()->hasFlexibleArrayMember())
        return ABIArgInfo::getIndirect(0);
    }
    
    // If specified, structs and unions are always indirect.
    if (!IsSmallStructInRegABI && !RetTy->isAnyComplexType())
      return ABIArgInfo::getIndirect(0);

    // Classify "single element" structs as their element type.
    if (const Type *SeltTy = isSingleElementStruct(RetTy, Context)) {
      if (const BuiltinType *BT = SeltTy->getAs<BuiltinType>()) {
        if (BT->isIntegerType()) {
          // We need to use the size of the structure, padding
          // bit-fields can adjust that to be larger than the single
          // element type.
          uint64_t Size = Context.getTypeSize(RetTy);
          return ABIArgInfo::getCoerce(
            llvm::IntegerType::get(VMContext, (unsigned) Size));
        } else if (BT->getKind() == BuiltinType::Float) {
          assert(Context.getTypeSize(RetTy) == Context.getTypeSize(SeltTy) &&
                 "Unexpect single element structure size!");
          return ABIArgInfo::getCoerce(llvm::Type::getFloatTy(VMContext));
        } else if (BT->getKind() == BuiltinType::Double) {
          assert(Context.getTypeSize(RetTy) == Context.getTypeSize(SeltTy) &&
                 "Unexpect single element structure size!");
          return ABIArgInfo::getCoerce(llvm::Type::getDoubleTy(VMContext));
        }
      } else if (SeltTy->isPointerType()) {
        // FIXME: It would be really nice if this could come out as the proper
        // pointer type.
        const llvm::Type *PtrTy = llvm::Type::getInt8PtrTy(VMContext);
        return ABIArgInfo::getCoerce(PtrTy);
      } else if (SeltTy->isVectorType()) {
        // 64- and 128-bit vectors are never returned in a
        // register when inside a structure.
        uint64_t Size = Context.getTypeSize(RetTy);
        if (Size == 64 || Size == 128)
          return ABIArgInfo::getIndirect(0);

        return classifyReturnType(QualType(SeltTy, 0), Context, VMContext);
      }
    }

    // Small structures which are register sized are generally returned
    // in a register.
    if (X86_32ABIInfo::shouldReturnTypeInRegister(RetTy, Context)) {
      uint64_t Size = Context.getTypeSize(RetTy);
      return ABIArgInfo::getCoerce(llvm::IntegerType::get(VMContext, Size));
    }

    return ABIArgInfo::getIndirect(0);
  } else {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }
}

unsigned X86_32ABIInfo::getIndirectArgumentAlignment(QualType Ty,
                                                     ASTContext &Context) {
  unsigned Align = Context.getTypeAlign(Ty);
  if (Align < 128) return 0;
  if (const RecordType* RT = Ty->getAs<RecordType>())
    if (typeContainsSSEVector(RT->getDecl(), Context))
      return 16;
  return 0;
}

ABIArgInfo X86_32ABIInfo::classifyArgumentType(QualType Ty,
                                               ASTContext &Context,
                                           llvm::LLVMContext &VMContext) const {
  // FIXME: Set alignment on indirect arguments.
  if (CodeGenFunction::hasAggregateLLVMType(Ty)) {
    // Structures with flexible arrays are always indirect.
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      // Structures with either a non-trivial destructor or a non-trivial
      // copy constructor are always indirect.
      if (hasNonTrivialDestructorOrCopyConstructor(RT))
        return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
        
      if (RT->getDecl()->hasFlexibleArrayMember())
        return ABIArgInfo::getIndirect(getIndirectArgumentAlignment(Ty,
                                                                    Context));
    }

    // Ignore empty structs.
    if (Ty->isStructureType() && Context.getTypeSize(Ty) == 0)
      return ABIArgInfo::getIgnore();

    // Expand small (<= 128-bit) record types when we know that the stack layout
    // of those arguments will match the struct. This is important because the
    // LLVM backend isn't smart enough to remove byval, which inhibits many
    // optimizations.
    if (Context.getTypeSize(Ty) <= 4*32 &&
        canExpandIndirectArgument(Ty, Context))
      return ABIArgInfo::getExpand();

    return ABIArgInfo::getIndirect(getIndirectArgumentAlignment(Ty, Context));
  } else {
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }
}

llvm::Value *X86_32ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  const llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, 4);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(
                          llvm::Type::getInt32Ty(CGF.getLLVMContext()), Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}

void X86_32TargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                                  llvm::GlobalValue *GV,
                                            CodeGen::CodeGenModule &CGM) const {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->hasAttr<X86ForceAlignArgPointerAttr>()) {
      // Get the LLVM function.
      llvm::Function *Fn = cast<llvm::Function>(GV);

      // Now add the 'alignstack' attribute with a value of 16.
      Fn->addFnAttr(llvm::Attribute::constructStackAlignmentFromInt(16));
    }
  }
}

bool X86_32TargetCodeGenInfo::initDwarfEHRegSizeTable(
                                               CodeGen::CodeGenFunction &CGF,
                                               llvm::Value *Address) const {
  CodeGen::CGBuilderTy &Builder = CGF.Builder;
  llvm::LLVMContext &Context = CGF.getLLVMContext();

  const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);
    
  // 0-7 are the eight integer registers;  the order is different
  //   on Darwin (for EH), but the range is the same.
  // 8 is %eip.
  for (unsigned I = 0, E = 9; I != E; ++I) {
    llvm::Value *Slot = Builder.CreateConstInBoundsGEP1_32(Address, I);
    Builder.CreateStore(Four8, Slot);
  }

  if (CGF.CGM.isTargetDarwin()) {
    // 12-16 are st(0..4).  Not sure why we stop at 4.
    // These have size 16, which is sizeof(long double) on
    // platforms with 8-byte alignment for that type.
    llvm::Value *Sixteen8 = llvm::ConstantInt::get(i8, 16);
    for (unsigned I = 12, E = 17; I != E; ++I) {
      llvm::Value *Slot = Builder.CreateConstInBoundsGEP1_32(Address, I);
      Builder.CreateStore(Sixteen8, Slot);
    }
      
  } else {
    // 9 is %eflags, which doesn't get a size on Darwin for some
    // reason.
    Builder.CreateStore(Four8, Builder.CreateConstInBoundsGEP1_32(Address, 9));

    // 11-16 are st(0..5).  Not sure why we stop at 5.
    // These have size 12, which is sizeof(long double) on
    // platforms with 4-byte alignment for that type.
    llvm::Value *Twelve8 = llvm::ConstantInt::get(i8, 12);
    for (unsigned I = 11, E = 17; I != E; ++I) {
      llvm::Value *Slot = Builder.CreateConstInBoundsGEP1_32(Address, I);
      Builder.CreateStore(Twelve8, Slot);
    }
  }      

  return false;
}

namespace {
/// X86_64ABIInfo - The X86_64 ABI information.
class X86_64ABIInfo : public ABIInfo {
  enum Class {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    NoClass,
    Memory
  };

  /// merge - Implement the X86_64 ABI merging algorithm.
  ///
  /// Merge an accumulating classification \arg Accum with a field
  /// classification \arg Field.
  ///
  /// \param Accum - The accumulating classification. This should
  /// always be either NoClass or the result of a previous merge
  /// call. In addition, this should never be Memory (the caller
  /// should just return Memory for the aggregate).
  Class merge(Class Accum, Class Field) const;

  /// classify - Determine the x86_64 register classes in which the
  /// given type T should be passed.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the
  /// containing object.  Some parameters are classified different
  /// depending on whether they straddle an eightbyte boundary.
  ///
  /// If a word is unused its result will be NoClass; if a type should
  /// be passed in Memory then at least the classification of \arg Lo
  /// will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will
  /// also be ComplexX87.
  void classify(QualType T, ASTContext &Context, uint64_t OffsetBase,
                Class &Lo, Class &Hi) const;

  /// getCoerceResult - Given a source type \arg Ty and an LLVM type
  /// to coerce to, chose the best way to pass Ty in the same place
  /// that \arg CoerceTo would be passed, but while keeping the
  /// emitted code as simple as possible.
  ///
  /// FIXME: Note, this should be cleaned up to just take an enumeration of all
  /// the ways we might want to pass things, instead of constructing an LLVM
  /// type. This makes this code more explicit, and it makes it clearer that we
  /// are also doing this for correctness in the case of passing scalar types.
  ABIArgInfo getCoerceResult(QualType Ty,
                             const llvm::Type *CoerceTo,
                             ASTContext &Context) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ABIArgInfo getIndirectResult(QualType Ty,
                               ASTContext &Context) const;

  ABIArgInfo classifyReturnType(QualType RetTy,
                                ASTContext &Context,
                                llvm::LLVMContext &VMContext) const;

  ABIArgInfo classifyArgumentType(QualType Ty,
                                  ASTContext &Context,
                                  llvm::LLVMContext &VMContext,
                                  unsigned &neededInt,
                                  unsigned &neededSSE) const;

public:
  virtual void computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                           llvm::LLVMContext &VMContext) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class X86_64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_64TargetCodeGenInfo():TargetCodeGenInfo(new X86_64ABIInfo()) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    CodeGen::CGBuilderTy &Builder = CGF.Builder;
    llvm::LLVMContext &Context = CGF.getLLVMContext();

    const llvm::IntegerType *i8 = llvm::Type::getInt8Ty(Context);
    llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);
      
    // 0-16 are the 16 integer registers.
    // 17 is %rip.
    for (unsigned I = 0, E = 17; I != E; ++I) {
      llvm::Value *Slot = Builder.CreateConstInBoundsGEP1_32(Address, I);
      Builder.CreateStore(Eight8, Slot);
    }

    return false;
  }
};

}

X86_64ABIInfo::Class X86_64ABIInfo::merge(Class Accum,
                                          Class Field) const {
  // AMD64-ABI 3.2.3p2: Rule 4. Each field of an object is
  // classified recursively so that always two fields are
  // considered. The resulting class is calculated according to
  // the classes of the fields in the eightbyte:
  //
  // (a) If both classes are equal, this is the resulting class.
  //
  // (b) If one of the classes is NO_CLASS, the resulting class is
  // the other class.
  //
  // (c) If one of the classes is MEMORY, the result is the MEMORY
  // class.
  //
  // (d) If one of the classes is INTEGER, the result is the
  // INTEGER.
  //
  // (e) If one of the classes is X87, X87UP, COMPLEX_X87 class,
  // MEMORY is used as class.
  //
  // (f) Otherwise class SSE is used.

  // Accum should never be memory (we should have returned) or
  // ComplexX87 (because this cannot be passed in a structure).
  assert((Accum != Memory && Accum != ComplexX87) &&
         "Invalid accumulated classification during merge.");
  if (Accum == Field || Field == NoClass)
    return Accum;
  else if (Field == Memory)
    return Memory;
  else if (Accum == NoClass)
    return Field;
  else if (Accum == Integer || Field == Integer)
    return Integer;
  else if (Field == X87 || Field == X87Up || Field == ComplexX87 ||
           Accum == X87 || Accum == X87Up)
    return Memory;
  else
    return SSE;
}

void X86_64ABIInfo::classify(QualType Ty,
                             ASTContext &Context,
                             uint64_t OffsetBase,
                             Class &Lo, Class &Hi) const {
  // FIXME: This code can be simplified by introducing a simple value class for
  // Class pairs with appropriate constructor methods for the various
  // situations.

  // FIXME: Some of the split computations are wrong; unaligned vectors
  // shouldn't be passed in registers for example, so there is no chance they
  // can straddle an eightbyte. Verify & simplify.

  Lo = Hi = NoClass;

  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Memory;

  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    BuiltinType::Kind k = BT->getKind();

    if (k == BuiltinType::Void) {
      Current = NoClass;
    } else if (k == BuiltinType::Int128 || k == BuiltinType::UInt128) {
      Lo = Integer;
      Hi = Integer;
    } else if (k >= BuiltinType::Bool && k <= BuiltinType::LongLong) {
      Current = Integer;
    } else if (k == BuiltinType::Float || k == BuiltinType::Double) {
      Current = SSE;
    } else if (k == BuiltinType::LongDouble) {
      Lo = X87;
      Hi = X87Up;
    }
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
  } else if (const EnumType *ET = Ty->getAs<EnumType>()) {
    // Classify the underlying integer type.
    classify(ET->getDecl()->getIntegerType(), Context, OffsetBase, Lo, Hi);
  } else if (Ty->hasPointerRepresentation()) {
    Current = Integer;
  } else if (const VectorType *VT = Ty->getAs<VectorType>()) {
    uint64_t Size = Context.getTypeSize(VT);
    if (Size == 32) {
      // gcc passes all <4 x char>, <2 x short>, <1 x int>, <1 x
      // float> as integer.
      Current = Integer;

      // If this type crosses an eightbyte boundary, it should be
      // split.
      uint64_t EB_Real = (OffsetBase) / 64;
      uint64_t EB_Imag = (OffsetBase + Size - 1) / 64;
      if (EB_Real != EB_Imag)
        Hi = Lo;
    } else if (Size == 64) {
      // gcc passes <1 x double> in memory. :(
      if (VT->getElementType()->isSpecificBuiltinType(BuiltinType::Double))
        return;

      // gcc passes <1 x long long> as INTEGER.
      if (VT->getElementType()->isSpecificBuiltinType(BuiltinType::LongLong))
        Current = Integer;
      else
        Current = SSE;

      // If this type crosses an eightbyte boundary, it should be
      // split.
      if (OffsetBase && OffsetBase != 64)
        Hi = Lo;
    } else if (Size == 128) {
      Lo = SSE;
      Hi = SSEUp;
    }
  } else if (const ComplexType *CT = Ty->getAs<ComplexType>()) {
    QualType ET = Context.getCanonicalType(CT->getElementType());

    uint64_t Size = Context.getTypeSize(Ty);
    if (ET->isIntegralType()) {
      if (Size <= 64)
        Current = Integer;
      else if (Size <= 128)
        Lo = Hi = Integer;
    } else if (ET == Context.FloatTy)
      Current = SSE;
    else if (ET == Context.DoubleTy)
      Lo = Hi = SSE;
    else if (ET == Context.LongDoubleTy)
      Current = ComplexX87;

    // If this complex type crosses an eightbyte boundary then it
    // should be split.
    uint64_t EB_Real = (OffsetBase) / 64;
    uint64_t EB_Imag = (OffsetBase + Context.getTypeSize(ET)) / 64;
    if (Hi == NoClass && EB_Real != EB_Imag)
      Hi = Lo;
  } else if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty)) {
    // Arrays are treated like structures.

    uint64_t Size = Context.getTypeSize(Ty);

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than two eightbytes, ..., it has class MEMORY.
    if (Size > 128)
      return;

    // AMD64-ABI 3.2.3p2: Rule 1. If ..., or it contains unaligned
    // fields, it has class MEMORY.
    //
    // Only need to check alignment of array base.
    if (OffsetBase % Context.getTypeAlign(AT->getElementType()))
      return;

    // Otherwise implement simplified merge. We could be smarter about
    // this, but it isn't worth it and would be harder to verify.
    Current = NoClass;
    uint64_t EltSize = Context.getTypeSize(AT->getElementType());
    uint64_t ArraySize = AT->getSize().getZExtValue();
    for (uint64_t i=0, Offset=OffsetBase; i<ArraySize; ++i, Offset += EltSize) {
      Class FieldLo, FieldHi;
      classify(AT->getElementType(), Context, Offset, FieldLo, FieldHi);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }

    // Do post merger cleanup (see below). Only case we worry about is Memory.
    if (Hi == Memory)
      Lo = Memory;
    assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp array classification.");
  } else if (const RecordType *RT = Ty->getAs<RecordType>()) {
    uint64_t Size = Context.getTypeSize(Ty);

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than two eightbytes, ..., it has class MEMORY.
    if (Size > 128)
      return;

    // AMD64-ABI 3.2.3p2: Rule 2. If a C++ object has either a non-trivial
    // copy constructor or a non-trivial destructor, it is passed by invisible
    // reference.
    if (hasNonTrivialDestructorOrCopyConstructor(RT))
      return;

    const RecordDecl *RD = RT->getDecl();

    // Assume variable sized types are passed in memory.
    if (RD->hasFlexibleArrayMember())
      return;

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    // Reset Lo class, this will be recomputed.
    Current = NoClass;

    // If this is a C++ record, classify the bases first.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
             e = CXXRD->bases_end(); i != e; ++i) {
        assert(!i->isVirtual() && !i->getType()->isDependentType() &&
               "Unexpected base class!");
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

        // Classify this field.
        //
        // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate exceeds a
        // single eightbyte, each is classified separately. Each eightbyte gets
        // initialized to class NO_CLASS.
        Class FieldLo, FieldHi;
        uint64_t Offset = OffsetBase + Layout.getBaseClassOffset(Base);
        classify(i->getType(), Context, Offset, FieldLo, FieldHi);
        Lo = merge(Lo, FieldLo);
        Hi = merge(Hi, FieldHi);
        if (Lo == Memory || Hi == Memory)
          break;
      }

      // If this record has no fields but isn't empty, classify as INTEGER.
      if (RD->field_empty() && Size)
        Current = Integer;
    }

    // Classify the fields one at a time, merging the results.
    unsigned idx = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
           i != e; ++i, ++idx) {
      uint64_t Offset = OffsetBase + Layout.getFieldOffset(idx);
      bool BitField = i->isBitField();

      // AMD64-ABI 3.2.3p2: Rule 1. If ..., or it contains unaligned
      // fields, it has class MEMORY.
      //
      // Note, skip this test for bit-fields, see below.
      if (!BitField && Offset % Context.getTypeAlign(i->getType())) {
        Lo = Memory;
        return;
      }

      // Classify this field.
      //
      // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate
      // exceeds a single eightbyte, each is classified
      // separately. Each eightbyte gets initialized to class
      // NO_CLASS.
      Class FieldLo, FieldHi;

      // Bit-fields require special handling, they do not force the
      // structure to be passed in memory even if unaligned, and
      // therefore they can straddle an eightbyte.
      if (BitField) {
        // Ignore padding bit-fields.
        if (i->isUnnamedBitfield())
          continue;

        uint64_t Offset = OffsetBase + Layout.getFieldOffset(idx);
        uint64_t Size = i->getBitWidth()->EvaluateAsInt(Context).getZExtValue();

        uint64_t EB_Lo = Offset / 64;
        uint64_t EB_Hi = (Offset + Size - 1) / 64;
        FieldLo = FieldHi = NoClass;
        if (EB_Lo) {
          assert(EB_Hi == EB_Lo && "Invalid classification, type > 16 bytes.");
          FieldLo = NoClass;
          FieldHi = Integer;
        } else {
          FieldLo = Integer;
          FieldHi = EB_Hi ? Integer : NoClass;
        }
      } else
        classify(i->getType(), Context, Offset, FieldLo, FieldHi);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }

    // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
    //
    // (a) If one of the classes is MEMORY, the whole argument is
    // passed in memory.
    //
    // (b) If SSEUP is not preceeded by SSE, it is converted to SSE.

    // The first of these conditions is guaranteed by how we implement
    // the merge (just bail).
    //
    // The second condition occurs in the case of unions; for example
    // union { _Complex double; unsigned; }.
    if (Hi == Memory)
      Lo = Memory;
    if (Hi == SSEUp && Lo != SSE)
      Hi = SSE;
  }
}

ABIArgInfo X86_64ABIInfo::getCoerceResult(QualType Ty,
                                          const llvm::Type *CoerceTo,
                                          ASTContext &Context) const {
  if (CoerceTo == llvm::Type::getInt64Ty(CoerceTo->getContext())) {
    // Integer and pointer types will end up in a general purpose
    // register.

    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    if (Ty->isIntegralType() || Ty->hasPointerRepresentation())
      return (Ty->isPromotableIntegerType() ?
              ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  } else if (CoerceTo == llvm::Type::getDoubleTy(CoerceTo->getContext())) {
    assert(Ty.isCanonical() && "should always have a canonical type here");
    assert(!Ty.hasQualifiers() && "should never have a qualified type here");

    // Float and double end up in a single SSE reg.
    if (Ty == Context.FloatTy || Ty == Context.DoubleTy)
      return ABIArgInfo::getDirect();

  }

  return ABIArgInfo::getCoerce(CoerceTo);
}

ABIArgInfo X86_64ABIInfo::getIndirectResult(QualType Ty,
                                            ASTContext &Context) const {
  // If this is a scalar LLVM value then assume LLVM will pass it in the right
  // place naturally.
  if (!CodeGenFunction::hasAggregateLLVMType(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  bool ByVal = !isRecordWithNonTrivialDestructorOrCopyConstructor(Ty);

  // FIXME: Set alignment correctly.
  return ABIArgInfo::getIndirect(0, ByVal);
}

ABIArgInfo X86_64ABIInfo::classifyReturnType(QualType RetTy,
                                            ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, Context, 0, Lo, Hi);

  // Check some invariants.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Lo != NoClass || Hi == NoClass) && "Invalid null classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  const llvm::Type *ResType = 0;
  switch (Lo) {
  case NoClass:
    return ABIArgInfo::getIgnore();

  case SSEUp:
  case X87Up:
    assert(0 && "Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p4: Rule 2. Types of class memory are returned via
    // hidden argument.
  case Memory:
    return getIndirectResult(RetTy, Context);

    // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next
    // available register of the sequence %rax, %rdx is used.
  case Integer:
    ResType = llvm::Type::getInt64Ty(VMContext); break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case SSE:
    ResType = llvm::Type::getDoubleTy(VMContext); break;

    // AMD64-ABI 3.2.3p4: Rule 6. If the class is X87, the value is
    // returned on the X87 stack in %st0 as 80-bit x87 number.
  case X87:
    ResType = llvm::Type::getX86_FP80Ty(VMContext); break;

    // AMD64-ABI 3.2.3p4: Rule 8. If the class is COMPLEX_X87, the real
    // part of the value is returned in %st0 and the imaginary part in
    // %st1.
  case ComplexX87:
    assert(Hi == ComplexX87 && "Unexpected ComplexX87 classification.");
    ResType = llvm::StructType::get(VMContext, llvm::Type::getX86_FP80Ty(VMContext),
                                    llvm::Type::getX86_FP80Ty(VMContext),
                                    NULL);
    break;
  }

  switch (Hi) {
    // Memory was handled previously and X87 should
    // never occur as a hi class.
  case Memory:
  case X87:
    assert(0 && "Invalid classification for hi word.");

  case ComplexX87: // Previously handled.
  case NoClass: break;

  case Integer:
    ResType = llvm::StructType::get(VMContext, ResType,
                                    llvm::Type::getInt64Ty(VMContext), NULL);
    break;
  case SSE:
    ResType = llvm::StructType::get(VMContext, ResType,
                                    llvm::Type::getDoubleTy(VMContext), NULL);
    break;

    // AMD64-ABI 3.2.3p4: Rule 5. If the class is SSEUP, the eightbyte
    // is passed in the upper half of the last used SSE register.
    //
    // SSEUP should always be preceeded by SSE, just widen.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification.");
    ResType = llvm::VectorType::get(llvm::Type::getDoubleTy(VMContext), 2);
    break;

    // AMD64-ABI 3.2.3p4: Rule 7. If the class is X87UP, the value is
    // returned together with the previous X87 value in %st0.
  case X87Up:
    // If X87Up is preceeded by X87, we don't need to do
    // anything. However, in some cases with unions it may not be
    // preceeded by X87. In such situations we follow gcc and pass the
    // extra bits in an SSE reg.
    if (Lo != X87)
      ResType = llvm::StructType::get(VMContext, ResType,
                                      llvm::Type::getDoubleTy(VMContext), NULL);
    break;
  }

  return getCoerceResult(RetTy, ResType, Context);
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(QualType Ty, ASTContext &Context,
                                               llvm::LLVMContext &VMContext,
                                               unsigned &neededInt,
                                               unsigned &neededSSE) const {
  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, Context, 0, Lo, Hi);

  // Check some invariants.
  // FIXME: Enforce these by construction.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Lo != NoClass || Hi == NoClass) && "Invalid null classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  const llvm::Type *ResType = 0;
  switch (Lo) {
  case NoClass:
    return ABIArgInfo::getIgnore();

    // AMD64-ABI 3.2.3p3: Rule 1. If the class is MEMORY, pass the argument
    // on the stack.
  case Memory:

    // AMD64-ABI 3.2.3p3: Rule 5. If the class is X87, X87UP or
    // COMPLEX_X87, it is passed in memory.
  case X87:
  case ComplexX87:
    return getIndirectResult(Ty, Context);

  case SSEUp:
  case X87Up:
    assert(0 && "Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next
    // available register of the sequence %rdi, %rsi, %rdx, %rcx, %r8
    // and %r9 is used.
  case Integer:
    ++neededInt;
    ResType = llvm::Type::getInt64Ty(VMContext);
    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next
    // available SSE register is used, the registers are taken in the
    // order from %xmm0 to %xmm7.
  case SSE:
    ++neededSSE;
    ResType = llvm::Type::getDoubleTy(VMContext);
    break;
  }

  switch (Hi) {
    // Memory was handled previously, ComplexX87 and X87 should
    // never occur as hi classes, and X87Up must be preceed by X87,
    // which is passed in memory.
  case Memory:
  case X87:
  case ComplexX87:
    assert(0 && "Invalid classification for hi word.");
    break;

  case NoClass: break;
  case Integer:
    ResType = llvm::StructType::get(VMContext, ResType,
                                    llvm::Type::getInt64Ty(VMContext), NULL);
    ++neededInt;
    break;

    // X87Up generally doesn't occur here (long double is passed in
    // memory), except in situations involving unions.
  case X87Up:
  case SSE:
    ResType = llvm::StructType::get(VMContext, ResType,
                                    llvm::Type::getDoubleTy(VMContext), NULL);
    ++neededSSE;
    break;

    // AMD64-ABI 3.2.3p3: Rule 4. If the class is SSEUP, the
    // eightbyte is passed in the upper half of the last used SSE
    // register.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification.");
    ResType = llvm::VectorType::get(llvm::Type::getDoubleTy(VMContext), 2);
    break;
  }

  return getCoerceResult(Ty, ResType, Context);
}

void X86_64ABIInfo::computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                                llvm::LLVMContext &VMContext) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType(),
                                          Context, VMContext);

  // Keep track of the number of assigned registers.
  unsigned freeIntRegs = 6, freeSSERegs = 8;

  // If the return value is indirect, then the hidden argument is consuming one
  // integer register.
  if (FI.getReturnInfo().isIndirect())
    --freeIntRegs;

  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it) {
    unsigned neededInt, neededSSE;
    it->info = classifyArgumentType(it->type, Context, VMContext,
                                    neededInt, neededSSE);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (freeIntRegs >= neededInt && freeSSERegs >= neededSSE) {
      freeIntRegs -= neededInt;
      freeSSERegs -= neededSSE;
    } else {
      it->info = getIndirectResult(it->type, Context);
    }
  }
}

static llvm::Value *EmitVAArgFromMemory(llvm::Value *VAListAddr,
                                        QualType Ty,
                                        CodeGenFunction &CGF) {
  llvm::Value *overflow_arg_area_p =
    CGF.Builder.CreateStructGEP(VAListAddr, 2, "overflow_arg_area_p");
  llvm::Value *overflow_arg_area =
    CGF.Builder.CreateLoad(overflow_arg_area_p, "overflow_arg_area");

  // AMD64-ABI 3.5.7p5: Step 7. Align l->overflow_arg_area upwards to a 16
  // byte boundary if alignment needed by type exceeds 8 byte boundary.
  uint64_t Align = CGF.getContext().getTypeAlign(Ty) / 8;
  if (Align > 8) {
    // Note that we follow the ABI & gcc here, even though the type
    // could in theory have an alignment greater than 16. This case
    // shouldn't ever matter in practice.

    // overflow_arg_area = (overflow_arg_area + 15) & ~15;
    llvm::Value *Offset =
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(CGF.getLLVMContext()), 15);
    overflow_arg_area = CGF.Builder.CreateGEP(overflow_arg_area, Offset);
    llvm::Value *AsInt = CGF.Builder.CreatePtrToInt(overflow_arg_area,
                                 llvm::Type::getInt64Ty(CGF.getLLVMContext()));
    llvm::Value *Mask = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(CGF.getLLVMContext()), ~15LL);
    overflow_arg_area =
      CGF.Builder.CreateIntToPtr(CGF.Builder.CreateAnd(AsInt, Mask),
                                 overflow_arg_area->getType(),
                                 "overflow_arg_area.align");
  }

  // AMD64-ABI 3.5.7p5: Step 8. Fetch type from l->overflow_arg_area.
  const llvm::Type *LTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *Res =
    CGF.Builder.CreateBitCast(overflow_arg_area,
                              llvm::PointerType::getUnqual(LTy));

  // AMD64-ABI 3.5.7p5: Step 9. Set l->overflow_arg_area to:
  // l->overflow_arg_area + sizeof(type).
  // AMD64-ABI 3.5.7p5: Step 10. Align l->overflow_arg_area upwards to
  // an 8 byte boundary.

  uint64_t SizeInBytes = (CGF.getContext().getTypeSize(Ty) + 7) / 8;
  llvm::Value *Offset =
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(CGF.getLLVMContext()),
                                               (SizeInBytes + 7)  & ~7);
  overflow_arg_area = CGF.Builder.CreateGEP(overflow_arg_area, Offset,
                                            "overflow_arg_area.next");
  CGF.Builder.CreateStore(overflow_arg_area, overflow_arg_area_p);

  // AMD64-ABI 3.5.7p5: Step 11. Return the fetched type.
  return Res;
}

llvm::Value *X86_64ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  llvm::LLVMContext &VMContext = CGF.getLLVMContext();
  const llvm::Type *i32Ty = llvm::Type::getInt32Ty(VMContext);
  const llvm::Type *DoubleTy = llvm::Type::getDoubleTy(VMContext);

  // Assume that va_list type is correct; should be pointer to LLVM type:
  // struct {
  //   i32 gp_offset;
  //   i32 fp_offset;
  //   i8* overflow_arg_area;
  //   i8* reg_save_area;
  // };
  unsigned neededInt, neededSSE;
  
  Ty = CGF.getContext().getCanonicalType(Ty);
  ABIArgInfo AI = classifyArgumentType(Ty, CGF.getContext(), VMContext,
                                       neededInt, neededSSE);

  // AMD64-ABI 3.5.7p5: Step 1. Determine whether type may be passed
  // in the registers. If not go to step 7.
  if (!neededInt && !neededSSE)
    return EmitVAArgFromMemory(VAListAddr, Ty, CGF);

  // AMD64-ABI 3.5.7p5: Step 2. Compute num_gp to hold the number of
  // general purpose registers needed to pass type and num_fp to hold
  // the number of floating point registers needed.

  // AMD64-ABI 3.5.7p5: Step 3. Verify whether arguments fit into
  // registers. In the case: l->gp_offset > 48 - num_gp * 8 or
  // l->fp_offset > 304 - num_fp * 16 go to step 7.
  //
  // NOTE: 304 is a typo, there are (6 * 8 + 8 * 16) = 176 bytes of
  // register save space).

  llvm::Value *InRegs = 0;
  llvm::Value *gp_offset_p = 0, *gp_offset = 0;
  llvm::Value *fp_offset_p = 0, *fp_offset = 0;
  if (neededInt) {
    gp_offset_p = CGF.Builder.CreateStructGEP(VAListAddr, 0, "gp_offset_p");
    gp_offset = CGF.Builder.CreateLoad(gp_offset_p, "gp_offset");
    InRegs =
      CGF.Builder.CreateICmpULE(gp_offset,
                                llvm::ConstantInt::get(i32Ty,
                                                       48 - neededInt * 8),
                                "fits_in_gp");
  }

  if (neededSSE) {
    fp_offset_p = CGF.Builder.CreateStructGEP(VAListAddr, 1, "fp_offset_p");
    fp_offset = CGF.Builder.CreateLoad(fp_offset_p, "fp_offset");
    llvm::Value *FitsInFP =
      CGF.Builder.CreateICmpULE(fp_offset,
                                llvm::ConstantInt::get(i32Ty,
                                                       176 - neededSSE * 16),
                                "fits_in_fp");
    InRegs = InRegs ? CGF.Builder.CreateAnd(InRegs, FitsInFP) : FitsInFP;
  }

  llvm::BasicBlock *InRegBlock = CGF.createBasicBlock("vaarg.in_reg");
  llvm::BasicBlock *InMemBlock = CGF.createBasicBlock("vaarg.in_mem");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("vaarg.end");
  CGF.Builder.CreateCondBr(InRegs, InRegBlock, InMemBlock);

  // Emit code to load the value if it was passed in registers.

  CGF.EmitBlock(InRegBlock);

  // AMD64-ABI 3.5.7p5: Step 4. Fetch type from l->reg_save_area with
  // an offset of l->gp_offset and/or l->fp_offset. This may require
  // copying to a temporary location in case the parameter is passed
  // in different register classes or requires an alignment greater
  // than 8 for general purpose registers and 16 for XMM registers.
  //
  // FIXME: This really results in shameful code when we end up needing to
  // collect arguments from different places; often what should result in a
  // simple assembling of a structure from scattered addresses has many more
  // loads than necessary. Can we clean this up?
  const llvm::Type *LTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *RegAddr =
    CGF.Builder.CreateLoad(CGF.Builder.CreateStructGEP(VAListAddr, 3),
                           "reg_save_area");
  if (neededInt && neededSSE) {
    // FIXME: Cleanup.
    assert(AI.isCoerce() && "Unexpected ABI info for mixed regs");
    const llvm::StructType *ST = cast<llvm::StructType>(AI.getCoerceToType());
    llvm::Value *Tmp = CGF.CreateTempAlloca(ST);
    assert(ST->getNumElements() == 2 && "Unexpected ABI info for mixed regs");
    const llvm::Type *TyLo = ST->getElementType(0);
    const llvm::Type *TyHi = ST->getElementType(1);
    assert((TyLo->isFloatingPointTy() ^ TyHi->isFloatingPointTy()) &&
           "Unexpected ABI info for mixed regs");
    const llvm::Type *PTyLo = llvm::PointerType::getUnqual(TyLo);
    const llvm::Type *PTyHi = llvm::PointerType::getUnqual(TyHi);
    llvm::Value *GPAddr = CGF.Builder.CreateGEP(RegAddr, gp_offset);
    llvm::Value *FPAddr = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    llvm::Value *RegLoAddr = TyLo->isFloatingPointTy() ? FPAddr : GPAddr;
    llvm::Value *RegHiAddr = TyLo->isFloatingPointTy() ? GPAddr : FPAddr;
    llvm::Value *V =
      CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegLoAddr, PTyLo));
    CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 0));
    V = CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegHiAddr, PTyHi));
    CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 1));

    RegAddr = CGF.Builder.CreateBitCast(Tmp,
                                        llvm::PointerType::getUnqual(LTy));
  } else if (neededInt) {
    RegAddr = CGF.Builder.CreateGEP(RegAddr, gp_offset);
    RegAddr = CGF.Builder.CreateBitCast(RegAddr,
                                        llvm::PointerType::getUnqual(LTy));
  } else {
    if (neededSSE == 1) {
      RegAddr = CGF.Builder.CreateGEP(RegAddr, fp_offset);
      RegAddr = CGF.Builder.CreateBitCast(RegAddr,
                                          llvm::PointerType::getUnqual(LTy));
    } else {
      assert(neededSSE == 2 && "Invalid number of needed registers!");
      // SSE registers are spaced 16 bytes apart in the register save
      // area, we need to collect the two eightbytes together.
      llvm::Value *RegAddrLo = CGF.Builder.CreateGEP(RegAddr, fp_offset);
      llvm::Value *RegAddrHi =
        CGF.Builder.CreateGEP(RegAddrLo,
                            llvm::ConstantInt::get(i32Ty, 16));
      const llvm::Type *DblPtrTy =
        llvm::PointerType::getUnqual(DoubleTy);
      const llvm::StructType *ST = llvm::StructType::get(VMContext, DoubleTy,
                                                         DoubleTy, NULL);
      llvm::Value *V, *Tmp = CGF.CreateTempAlloca(ST);
      V = CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegAddrLo,
                                                           DblPtrTy));
      CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 0));
      V = CGF.Builder.CreateLoad(CGF.Builder.CreateBitCast(RegAddrHi,
                                                           DblPtrTy));
      CGF.Builder.CreateStore(V, CGF.Builder.CreateStructGEP(Tmp, 1));
      RegAddr = CGF.Builder.CreateBitCast(Tmp,
                                          llvm::PointerType::getUnqual(LTy));
    }
  }

  // AMD64-ABI 3.5.7p5: Step 5. Set:
  // l->gp_offset = l->gp_offset + num_gp * 8
  // l->fp_offset = l->fp_offset + num_fp * 16.
  if (neededInt) {
    llvm::Value *Offset = llvm::ConstantInt::get(i32Ty, neededInt * 8);
    CGF.Builder.CreateStore(CGF.Builder.CreateAdd(gp_offset, Offset),
                            gp_offset_p);
  }
  if (neededSSE) {
    llvm::Value *Offset = llvm::ConstantInt::get(i32Ty, neededSSE * 16);
    CGF.Builder.CreateStore(CGF.Builder.CreateAdd(fp_offset, Offset),
                            fp_offset_p);
  }
  CGF.EmitBranch(ContBlock);

  // Emit code to load the value if it was passed in memory.

  CGF.EmitBlock(InMemBlock);
  llvm::Value *MemAddr = EmitVAArgFromMemory(VAListAddr, Ty, CGF);

  // Return the appropriate result.

  CGF.EmitBlock(ContBlock);
  llvm::PHINode *ResAddr = CGF.Builder.CreatePHI(RegAddr->getType(),
                                                 "vaarg.addr");
  ResAddr->reserveOperandSpace(2);
  ResAddr->addIncoming(RegAddr, InRegBlock);
  ResAddr->addIncoming(MemAddr, InMemBlock);

  return ResAddr;
}

// PIC16 ABI Implementation

namespace {

class PIC16ABIInfo : public ABIInfo {
  ABIArgInfo classifyReturnType(QualType RetTy,
                                ASTContext &Context,
                                llvm::LLVMContext &VMContext) const;

  ABIArgInfo classifyArgumentType(QualType RetTy,
                                  ASTContext &Context,
                                  llvm::LLVMContext &VMContext) const;

  virtual void computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                           llvm::LLVMContext &VMContext) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), Context,
                                            VMContext);
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type, Context, VMContext);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class PIC16TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  PIC16TargetCodeGenInfo():TargetCodeGenInfo(new PIC16ABIInfo()) {}
};

}

ABIArgInfo PIC16ABIInfo::classifyReturnType(QualType RetTy,
                                            ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  if (RetTy->isVoidType()) {
    return ABIArgInfo::getIgnore();
  } else {
    return ABIArgInfo::getDirect();
  }
}

ABIArgInfo PIC16ABIInfo::classifyArgumentType(QualType Ty,
                                              ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  return ABIArgInfo::getDirect();
}

llvm::Value *PIC16ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(CGF.getLLVMContext()));
  const llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset = CGF.getContext().getTypeSize(Ty) / 8;

  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(
                          llvm::Type::getInt32Ty(CGF.getLLVMContext()), Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}


// ARM ABI Implementation

namespace {

class ARMABIInfo : public ABIInfo {
public:
  enum ABIKind {
    APCS = 0,
    AAPCS = 1,
    AAPCS_VFP
  };

private:
  ABIKind Kind;

public:
  ARMABIInfo(ABIKind _Kind) : Kind(_Kind) {}

private:
  ABIKind getABIKind() const { return Kind; }

  ABIArgInfo classifyReturnType(QualType RetTy,
                                ASTContext &Context,
                                llvm::LLVMContext &VMCOntext) const;

  ABIArgInfo classifyArgumentType(QualType RetTy,
                                  ASTContext &Context,
                                  llvm::LLVMContext &VMContext) const;

  virtual void computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                           llvm::LLVMContext &VMContext) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class ARMTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  ARMTargetCodeGenInfo(ARMABIInfo::ABIKind K)
    :TargetCodeGenInfo(new ARMABIInfo(K)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    return 13;
  }
};

}

void ARMABIInfo::computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                             llvm::LLVMContext &VMContext) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), Context,
                                          VMContext);
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it) {
    it->info = classifyArgumentType(it->type, Context, VMContext);
  }

  // ARM always overrides the calling convention.
  switch (getABIKind()) {
  case APCS:
    FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_APCS);
    break;

  case AAPCS:
    FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_AAPCS);
    break;

  case AAPCS_VFP:
    FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_AAPCS_VFP);
    break;
  }
}

ABIArgInfo ARMABIInfo::classifyArgumentType(QualType Ty,
                                            ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  if (!CodeGenFunction::hasAggregateLLVMType(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Ignore empty records.
  if (isEmptyRecord(Context, Ty, true))
    return ABIArgInfo::getIgnore();

  // FIXME: This is kind of nasty... but there isn't much choice because the ARM
  // backend doesn't support byval.
  // FIXME: This doesn't handle alignment > 64 bits.
  const llvm::Type* ElemTy;
  unsigned SizeRegs;
  if (Context.getTypeAlign(Ty) > 32) {
    ElemTy = llvm::Type::getInt64Ty(VMContext);
    SizeRegs = (Context.getTypeSize(Ty) + 63) / 64;
  } else {
    ElemTy = llvm::Type::getInt32Ty(VMContext);
    SizeRegs = (Context.getTypeSize(Ty) + 31) / 32;
  }
  std::vector<const llvm::Type*> LLVMFields;
  LLVMFields.push_back(llvm::ArrayType::get(ElemTy, SizeRegs));
  const llvm::Type* STy = llvm::StructType::get(VMContext, LLVMFields, true);
  return ABIArgInfo::getCoerce(STy);
}

static bool isIntegerLikeType(QualType Ty,
                              ASTContext &Context,
                              llvm::LLVMContext &VMContext) {
  // APCS, C Language Calling Conventions, Non-Simple Return Values: A structure
  // is called integer-like if its size is less than or equal to one word, and
  // the offset of each of its addressable sub-fields is zero.

  uint64_t Size = Context.getTypeSize(Ty);

  // Check that the type fits in a word.
  if (Size > 32)
    return false;

  // FIXME: Handle vector types!
  if (Ty->isVectorType())
    return false;

  // Float types are never treated as "integer like".
  if (Ty->isRealFloatingType())
    return false;

  // If this is a builtin or pointer type then it is ok.
  if (Ty->getAs<BuiltinType>() || Ty->isPointerType())
    return true;

  // Small complex integer types are "integer like".
  if (const ComplexType *CT = Ty->getAs<ComplexType>())
    return isIntegerLikeType(CT->getElementType(), Context, VMContext);

  // Single element and zero sized arrays should be allowed, by the definition
  // above, but they are not.

  // Otherwise, it must be a record type.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT) return false;

  // Ignore records with flexible arrays.
  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return false;

  // Check that all sub-fields are at offset 0, and are themselves "integer
  // like".
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  bool HadField = false;
  unsigned idx = 0;
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
       i != e; ++i, ++idx) {
    const FieldDecl *FD = *i;

    // Bit-fields are not addressable, we only need to verify they are "integer
    // like". We still have to disallow a subsequent non-bitfield, for example:
    //   struct { int : 0; int x }
    // is non-integer like according to gcc.
    if (FD->isBitField()) {
      if (!RD->isUnion())
        HadField = true;

      if (!isIntegerLikeType(FD->getType(), Context, VMContext))
        return false;

      continue;
    }

    // Check if this field is at offset 0.
    if (Layout.getFieldOffset(idx) != 0)
      return false;

    if (!isIntegerLikeType(FD->getType(), Context, VMContext))
      return false;
    
    // Only allow at most one field in a structure. This doesn't match the
    // wording above, but follows gcc in situations with a field following an
    // empty structure.
    if (!RD->isUnion()) {
      if (HadField)
        return false;

      HadField = true;
    }
  }

  return true;
}

ABIArgInfo ARMABIInfo::classifyReturnType(QualType RetTy,
                                          ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (!CodeGenFunction::hasAggregateLLVMType(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Are we following APCS?
  if (getABIKind() == APCS) {
    if (isEmptyRecord(Context, RetTy, false))
      return ABIArgInfo::getIgnore();

    // Complex types are all returned as packed integers.
    //
    // FIXME: Consider using 2 x vector types if the back end handles them
    // correctly.
    if (RetTy->isAnyComplexType())
      return ABIArgInfo::getCoerce(llvm::IntegerType::get(
                                     VMContext, Context.getTypeSize(RetTy)));

    // Integer like structures are returned in r0.
    if (isIntegerLikeType(RetTy, Context, VMContext)) {
      // Return in the smallest viable integer type.
      uint64_t Size = Context.getTypeSize(RetTy);
      if (Size <= 8)
        return ABIArgInfo::getCoerce(llvm::Type::getInt8Ty(VMContext));
      if (Size <= 16)
        return ABIArgInfo::getCoerce(llvm::Type::getInt16Ty(VMContext));
      return ABIArgInfo::getCoerce(llvm::Type::getInt32Ty(VMContext));
    }

    // Otherwise return in memory.
    return ABIArgInfo::getIndirect(0);
  }

  // Otherwise this is an AAPCS variant.

  if (isEmptyRecord(Context, RetTy, true))
    return ABIArgInfo::getIgnore();

  // Aggregates <= 4 bytes are returned in r0; other aggregates
  // are returned indirectly.
  uint64_t Size = Context.getTypeSize(RetTy);
  if (Size <= 32) {
    // Return in the smallest viable integer type.
    if (Size <= 8)
      return ABIArgInfo::getCoerce(llvm::Type::getInt8Ty(VMContext));
    if (Size <= 16)
      return ABIArgInfo::getCoerce(llvm::Type::getInt16Ty(VMContext));
    return ABIArgInfo::getCoerce(llvm::Type::getInt32Ty(VMContext));
  }

  return ABIArgInfo::getIndirect(0);
}

llvm::Value *ARMABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  // FIXME: Need to handle alignment
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  const llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, 4);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(
                          llvm::Type::getInt32Ty(CGF.getLLVMContext()), Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}

ABIArgInfo DefaultABIInfo::classifyReturnType(QualType RetTy,
                                              ASTContext &Context,
                                          llvm::LLVMContext &VMContext) const {
  if (RetTy->isVoidType()) {
    return ABIArgInfo::getIgnore();
  } else if (CodeGenFunction::hasAggregateLLVMType(RetTy)) {
    return ABIArgInfo::getIndirect(0);
  } else {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }
}

// SystemZ ABI Implementation

namespace {

class SystemZABIInfo : public ABIInfo {
  bool isPromotableIntegerType(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy, ASTContext &Context,
                                llvm::LLVMContext &VMContext) const;

  ABIArgInfo classifyArgumentType(QualType RetTy, ASTContext &Context,
                                  llvm::LLVMContext &VMContext) const;

  virtual void computeInfo(CGFunctionInfo &FI, ASTContext &Context,
                          llvm::LLVMContext &VMContext) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType(),
                                            Context, VMContext);
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type, Context, VMContext);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class SystemZTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  SystemZTargetCodeGenInfo():TargetCodeGenInfo(new SystemZABIInfo()) {}
};

}

bool SystemZABIInfo::isPromotableIntegerType(QualType Ty) const {
  // SystemZ ABI requires all 8, 16 and 32 bit quantities to be extended.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Bool:
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return true;
    default:
      return false;
    }
  return false;
}

llvm::Value *SystemZABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  // FIXME: Implement
  return 0;
}


ABIArgInfo SystemZABIInfo::classifyReturnType(QualType RetTy,
                                              ASTContext &Context,
                                           llvm::LLVMContext &VMContext) const {
  if (RetTy->isVoidType()) {
    return ABIArgInfo::getIgnore();
  } else if (CodeGenFunction::hasAggregateLLVMType(RetTy)) {
    return ABIArgInfo::getIndirect(0);
  } else {
    return (isPromotableIntegerType(RetTy) ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }
}

ABIArgInfo SystemZABIInfo::classifyArgumentType(QualType Ty,
                                                ASTContext &Context,
                                           llvm::LLVMContext &VMContext) const {
  if (CodeGenFunction::hasAggregateLLVMType(Ty)) {
    return ABIArgInfo::getIndirect(0);
  } else {
    return (isPromotableIntegerType(Ty) ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }
}

// MSP430 ABI Implementation

namespace {

class MSP430TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  MSP430TargetCodeGenInfo():TargetCodeGenInfo(new DefaultABIInfo()) {}
  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const;
};

}

void MSP430TargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                                  llvm::GlobalValue *GV,
                                             CodeGen::CodeGenModule &M) const {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (const MSP430InterruptAttr *attr = FD->getAttr<MSP430InterruptAttr>()) {
      // Handle 'interrupt' attribute:
      llvm::Function *F = cast<llvm::Function>(GV);

      // Step 1: Set ISR calling convention.
      F->setCallingConv(llvm::CallingConv::MSP430_INTR);

      // Step 2: Add attributes goodness.
      F->addFnAttr(llvm::Attribute::NoInline);

      // Step 3: Emit ISR vector alias.
      unsigned Num = attr->getNumber() + 0xffe0;
      new llvm::GlobalAlias(GV->getType(), llvm::Function::ExternalLinkage,
                            "vector_" +
                            llvm::LowercaseString(llvm::utohexstr(Num)),
                            GV, &M.getModule());
    }
  }
}

const TargetCodeGenInfo &CodeGenModule::getTargetCodeGenInfo() const {
  if (TheTargetCodeGenInfo)
    return *TheTargetCodeGenInfo;

  // For now we just cache the TargetCodeGenInfo in CodeGenModule and don't
  // free it.

  const llvm::Triple &Triple(getContext().Target.getTriple());
  switch (Triple.getArch()) {
  default:
    return *(TheTargetCodeGenInfo = new DefaultTargetCodeGenInfo);

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // FIXME: We want to know the float calling convention as well.
    if (strcmp(getContext().Target.getABI(), "apcs-gnu") == 0)
      return *(TheTargetCodeGenInfo =
               new ARMTargetCodeGenInfo(ARMABIInfo::APCS));

    return *(TheTargetCodeGenInfo =
             new ARMTargetCodeGenInfo(ARMABIInfo::AAPCS));

  case llvm::Triple::pic16:
    return *(TheTargetCodeGenInfo = new PIC16TargetCodeGenInfo());

  case llvm::Triple::systemz:
    return *(TheTargetCodeGenInfo = new SystemZTargetCodeGenInfo());

  case llvm::Triple::msp430:
    return *(TheTargetCodeGenInfo = new MSP430TargetCodeGenInfo());

  case llvm::Triple::x86:
    switch (Triple.getOS()) {
    case llvm::Triple::Darwin:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Context, true, true));
    case llvm::Triple::Cygwin:
    case llvm::Triple::MinGW32:
    case llvm::Triple::MinGW64:
    case llvm::Triple::AuroraUX:
    case llvm::Triple::DragonFly:
    case llvm::Triple::FreeBSD:
    case llvm::Triple::OpenBSD:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Context, false, true));

    default:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Context, false, false));
    }

  case llvm::Triple::x86_64:
    return *(TheTargetCodeGenInfo = new X86_64TargetCodeGenInfo());
  }
}
