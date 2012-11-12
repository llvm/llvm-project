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
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Type.h"
#include "llvm/DataLayout.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace CodeGen;

static void AssignToArrayRange(CodeGen::CGBuilderTy &Builder,
                               llvm::Value *Array,
                               llvm::Value *Value,
                               unsigned FirstIndex,
                               unsigned LastIndex) {
  // Alternatively, we could emit this as a loop in the source.
  for (unsigned I = FirstIndex; I <= LastIndex; ++I) {
    llvm::Value *Cell = Builder.CreateConstInBoundsGEP1_32(Array, I);
    Builder.CreateStore(Value, Cell);
  }
}

static bool isAggregateTypeForABI(QualType T) {
  return CodeGenFunction::hasAggregateLLVMType(T) ||
         T->isMemberFunctionPointerType();
}

ABIInfo::~ABIInfo() {}

ASTContext &ABIInfo::getContext() const {
  return CGT.getContext();
}

llvm::LLVMContext &ABIInfo::getVMContext() const {
  return CGT.getLLVMContext();
}

const llvm::DataLayout &ABIInfo::getDataLayout() const {
  return CGT.getDataLayout();
}


void ABIArgInfo::dump() const {
  raw_ostream &OS = llvm::errs();
  OS << "(ABIArgInfo Kind=";
  switch (TheKind) {
  case Direct:
    OS << "Direct Type=";
    if (llvm::Type *Ty = getCoerceToType())
      Ty->print(OS);
    else
      OS << "null";
    break;
  case Extend:
    OS << "Extend";
    break;
  case Ignore:
    OS << "Ignore";
    break;
  case Indirect:
    OS << "Indirect Align=" << getIndirectAlign()
       << " ByVal=" << getIndirectByVal()
       << " Realign=" << getIndirectRealign();
    break;
  case Expand:
    OS << "Expand";
    break;
  }
  OS << ")\n";
}

TargetCodeGenInfo::~TargetCodeGenInfo() { delete Info; }

// If someone can figure out a general rule for this, that would be great.
// It's probably just doomed to be platform-dependent, though.
unsigned TargetCodeGenInfo::getSizeOfUnwindException() const {
  // Verified for:
  //   x86-64     FreeBSD, Linux, Darwin
  //   x86-32     FreeBSD, Linux, Darwin
  //   PowerPC    Linux, Darwin
  //   ARM        Darwin (*not* EABI)
  return 32;
}

bool TargetCodeGenInfo::isNoProtoCallVariadic(const CallArgList &args,
                                     const FunctionNoProtoType *fnType) const {
  // The following conventions are known to require this to be false:
  //   x86_stdcall
  //   MIPS
  // For everything else, we just prefer false unless we opt out.
  return false;
}

static bool isEmptyRecord(ASTContext &Context, QualType T, bool AllowArrays);

/// isEmptyField - Return true iff a the field is "empty", that is it
/// is an unnamed bit-field or an (array of) empty record(s).
static bool isEmptyField(ASTContext &Context, const FieldDecl *FD,
                         bool AllowArrays) {
  if (FD->isUnnamedBitfield())
    return true;

  QualType FT = FD->getType();

  // Constant arrays of empty records count as empty, strip them off.
  // Constant arrays of zero length always count as empty.
  if (AllowArrays)
    while (const ConstantArrayType *AT = Context.getAsConstantArrayType(FT)) {
      if (AT->getSize() == 0)
        return true;
      FT = AT->getElementType();
    }

  const RecordType *RT = FT->getAs<RecordType>();
  if (!RT)
    return false;

  // C++ record fields are never empty, at least in the Itanium ABI.
  //
  // FIXME: We should use a predicate for whether this behavior is true in the
  // current ABI.
  if (isa<CXXRecordDecl>(RT->getDecl()))
    return false;

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

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i)
      if (!isEmptyRecord(Context, i->getType(), true))
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

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i) {
      // Ignore empty records.
      if (isEmptyRecord(Context, i->getType(), true))
        continue;

      // If we already found an element then this isn't a single-element struct.
      if (Found)
        return 0;

      // If this is non-empty and not a single element struct, the composite
      // cannot be a single element struct.
      Found = isSingleElementStruct(i->getType(), Context);
      if (!Found)
        return 0;
    }
  }

  // Check for single element.
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

    if (!isAggregateTypeForABI(FT)) {
      Found = FT.getTypePtr();
    } else {
      Found = isSingleElementStruct(FT, Context);
      if (!Found)
        return 0;
    }
  }

  // We don't consider a struct a single-element struct if it has
  // padding beyond the element type.
  if (Found && Context.getTypeSize(Found) != Context.getTypeSize(T))
    return 0;

  return Found;
}

static bool is32Or64BitBasicType(QualType Ty, ASTContext &Context) {
  if (!Ty->getAs<BuiltinType>() && !Ty->hasPointerRepresentation() &&
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

  uint64_t Size = 0;

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

    Size += Context.getTypeSize(FD->getType());
  }

  // Make sure there are not any holes in the struct.
  if (Size != Context.getTypeSize(Ty))
    return false;

  return true;
}

namespace {
/// DefaultABIInfo - The default implementation for ABI specific
/// details. This implementation provides information which results in
/// self-consistent and sensible LLVM IR generation, but does not
/// conform to any particular ABI.
class DefaultABIInfo : public ABIInfo {
public:
  DefaultABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class DefaultTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  DefaultTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}
};

llvm::Value *DefaultABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  return 0;
}

ABIArgInfo DefaultABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty)) {
    // Records with non trivial destructors/constructors should not be passed
    // by value.
    if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    return ABIArgInfo::getIndirect(0);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  return (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo DefaultABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

//===----------------------------------------------------------------------===//
// le32/PNaCl bitcode ABI Implementation
//===----------------------------------------------------------------------===//

class PNaClABIInfo : public ABIInfo {
 public:
  PNaClABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, unsigned &FreeRegs) const;

  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class PNaClTargetCodeGenInfo : public TargetCodeGenInfo {
 public:
  PNaClTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new PNaClABIInfo(CGT)) {}
};

void PNaClABIInfo::computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    unsigned FreeRegs = FI.getHasRegParm() ? FI.getRegParm() : 0;

    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type, FreeRegs);
  }

llvm::Value *PNaClABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  return 0;
}

ABIArgInfo PNaClABIInfo::classifyArgumentType(QualType Ty,
                                              unsigned &FreeRegs) const {
  if (isAggregateTypeForABI(Ty)) {
    // Records with non trivial destructors/constructors should not be passed
    // by value.
    FreeRegs = 0;
    if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    return ABIArgInfo::getIndirect(0);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  ABIArgInfo BaseInfo = (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());

  // Regparm regs hold 32 bits.
  unsigned SizeInRegs = (getContext().getTypeSize(Ty) + 31) / 32;
  if (SizeInRegs == 0) return BaseInfo;
  if (SizeInRegs > FreeRegs) {
    FreeRegs = 0;
    return BaseInfo;
  }
  FreeRegs -= SizeInRegs;
  return BaseInfo.isDirect() ?
      ABIArgInfo::getDirectInReg(BaseInfo.getCoerceToType()) :
      ABIArgInfo::getExtendInReg(BaseInfo.getCoerceToType());
}

ABIArgInfo PNaClABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

/// UseX86_MMXType - Return true if this is an MMX type that should use the
/// special x86_mmx type.
bool UseX86_MMXType(llvm::Type *IRType) {
  // If the type is an MMX type <2 x i32>, <4 x i16>, or <8 x i8>, use the
  // special x86_mmx type.
  return IRType->isVectorTy() && IRType->getPrimitiveSizeInBits() == 64 &&
    cast<llvm::VectorType>(IRType)->getElementType()->isIntegerTy() &&
    IRType->getScalarSizeInBits() != 64;
}

static llvm::Type* X86AdjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                          StringRef Constraint,
                                          llvm::Type* Ty) {
  if ((Constraint == "y" || Constraint == "&y") && Ty->isVectorTy())
    return llvm::Type::getX86_MMXTy(CGF.getLLVMContext());
  return Ty;
}

//===----------------------------------------------------------------------===//
// X86-32 ABI Implementation
//===----------------------------------------------------------------------===//

/// X86_32ABIInfo - The X86-32 ABI information.
class X86_32ABIInfo : public ABIInfo {
  enum Class {
    Integer,
    Float
  };

  static const unsigned MinABIStackAlignInBytes = 4;

  bool IsDarwinVectorABI;
  bool IsSmallStructInRegABI;
  bool IsMMXDisabled;
  bool IsWin32FloatStructABI;
  unsigned DefaultNumRegisterParameters;

  static bool isRegisterSize(unsigned Size) {
    return (Size == 8 || Size == 16 || Size == 32 || Size == 64);
  }

  static bool shouldReturnTypeInRegister(QualType Ty, ASTContext &Context, 
                                          unsigned callingConvention);

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ABIArgInfo getIndirectResult(QualType Ty, bool ByVal,
                               unsigned &FreeRegs) const;

  /// \brief Return the alignment to use for the given type on the stack.
  unsigned getTypeStackAlignInBytes(QualType Ty, unsigned Align) const;

  Class classify(QualType Ty) const;
  ABIArgInfo classifyReturnType(QualType RetTy,
                                unsigned callingConvention) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, unsigned &FreeRegs,
                                  bool IsFastCall) const;
  bool shouldUseInReg(QualType Ty, unsigned &FreeRegs,
                      bool IsFastCall, bool &NeedsPadding) const;

public:

  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;

  X86_32ABIInfo(CodeGen::CodeGenTypes &CGT, bool d, bool p, bool m, bool w,
                unsigned r)
    : ABIInfo(CGT), IsDarwinVectorABI(d), IsSmallStructInRegABI(p),
      IsMMXDisabled(m), IsWin32FloatStructABI(w),
      DefaultNumRegisterParameters(r) {}
};

class X86_32TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_32TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT,
      bool d, bool p, bool m, bool w, unsigned r)
    :TargetCodeGenInfo(new X86_32ABIInfo(CGT, d, p, m, w, r)) {}

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const;

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    // Darwin uses different dwarf register numbers for EH.
    if (CGM.isTargetDarwin()) return 5;

    return 4;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;

  llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                  StringRef Constraint,
                                  llvm::Type* Ty) const {
    return X86AdjustInlineAsmType(CGF, Constraint, Ty);
  }

};

}

/// shouldReturnTypeInRegister - Determine if the given type should be
/// passed in a register (for the Darwin ABI).
bool X86_32ABIInfo::shouldReturnTypeInRegister(QualType Ty,
                                               ASTContext &Context,
                                               unsigned callingConvention) {
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

  // If this is a builtin, pointer, enum, complex type, member pointer, or
  // member function pointer it is ok.
  if (Ty->getAs<BuiltinType>() || Ty->hasPointerRepresentation() ||
      Ty->isAnyComplexType() || Ty->isEnumeralType() ||
      Ty->isBlockPointerType() || Ty->isMemberPointerType())
    return true;

  // Arrays are treated like records.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty))
    return shouldReturnTypeInRegister(AT->getElementType(), Context,
                                      callingConvention);

  // Otherwise, it must be a record type.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT) return false;

  // FIXME: Traverse bases here too.

  // For thiscall conventions, structures will never be returned in
  // a register.  This is for compatibility with the MSVC ABI
  if (callingConvention == llvm::CallingConv::X86_ThisCall && 
      RT->isStructureType()) {
    return false;
  }

  // Structure types are passed in register if all fields would be
  // passed in a register.
  for (RecordDecl::field_iterator i = RT->getDecl()->field_begin(),
         e = RT->getDecl()->field_end(); i != e; ++i) {
    const FieldDecl *FD = *i;

    // Empty fields are ignored.
    if (isEmptyField(Context, FD, true))
      continue;

    // Check fields recursively.
    if (!shouldReturnTypeInRegister(FD->getType(), Context, 
                                    callingConvention))
      return false;
  }
  return true;
}

ABIArgInfo X86_32ABIInfo::classifyReturnType(QualType RetTy, 
                                            unsigned callingConvention) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (const VectorType *VT = RetTy->getAs<VectorType>()) {
    // On Darwin, some vectors are returned in registers.
    if (IsDarwinVectorABI) {
      uint64_t Size = getContext().getTypeSize(RetTy);

      // 128-bit vectors are a special case; they are returned in
      // registers and we need to make sure to pick a type the LLVM
      // backend will like.
      if (Size == 128)
        return ABIArgInfo::getDirect(llvm::VectorType::get(
                  llvm::Type::getInt64Ty(getVMContext()), 2));

      // Always return in register if it fits in a general purpose
      // register, or if it is 64 bits and has a single element.
      if ((Size == 8 || Size == 16 || Size == 32) ||
          (Size == 64 && VT->getNumElements() == 1))
        return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                            Size));

      return ABIArgInfo::getIndirect(0);
    }

    return ABIArgInfo::getDirect();
  }

  if (isAggregateTypeForABI(RetTy)) {
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

    // Small structures which are register sized are generally returned
    // in a register.
    if (X86_32ABIInfo::shouldReturnTypeInRegister(RetTy, getContext(), 
                                                  callingConvention)) {
      uint64_t Size = getContext().getTypeSize(RetTy);

      // As a special-case, if the struct is a "single-element" struct, and
      // the field is of type "float" or "double", return it in a
      // floating-point register. (MSVC does not apply this special case.)
      // We apply a similar transformation for pointer types to improve the
      // quality of the generated IR.
      if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
        if ((!IsWin32FloatStructABI && SeltTy->isRealFloatingType())
            || SeltTy->hasPointerRepresentation())
          return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

      // FIXME: We should be able to narrow this integer in cases with dead
      // padding.
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),Size));
    }

    return ABIArgInfo::getIndirect(0);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

static bool isSSEVectorType(ASTContext &Context, QualType Ty) {
  return Ty->getAs<VectorType>() && Context.getTypeSize(Ty) == 128;
}

static bool isRecordWithSSEVectorType(ASTContext &Context, QualType Ty) {
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT)
    return 0;
  const RecordDecl *RD = RT->getDecl();

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i)
      if (!isRecordWithSSEVectorType(Context, i->getType()))
        return false;

  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
       i != e; ++i) {
    QualType FT = i->getType();

    if (isSSEVectorType(Context, FT))
      return true;

    if (isRecordWithSSEVectorType(Context, FT))
      return true;
  }

  return false;
}

unsigned X86_32ABIInfo::getTypeStackAlignInBytes(QualType Ty,
                                                 unsigned Align) const {
  // Otherwise, if the alignment is less than or equal to the minimum ABI
  // alignment, just use the default; the backend will handle this.
  if (Align <= MinABIStackAlignInBytes)
    return 0; // Use default alignment.

  // On non-Darwin, the stack type alignment is always 4.
  if (!IsDarwinVectorABI) {
    // Set explicit alignment, since we may need to realign the top.
    return MinABIStackAlignInBytes;
  }

  // Otherwise, if the type contains an SSE vector type, the alignment is 16.
  if (Align >= 16 && (isSSEVectorType(getContext(), Ty) ||
                      isRecordWithSSEVectorType(getContext(), Ty)))
    return 16;

  return MinABIStackAlignInBytes;
}

ABIArgInfo X86_32ABIInfo::getIndirectResult(QualType Ty, bool ByVal,
                                            unsigned &FreeRegs) const {
  if (!ByVal) {
    if (FreeRegs) {
      --FreeRegs; // Non byval indirects just use one pointer.
      return ABIArgInfo::getIndirectInReg(0, false);
    }
    return ABIArgInfo::getIndirect(0, false);
  }

  // Compute the byval alignment.
  unsigned TypeAlign = getContext().getTypeAlign(Ty) / 8;
  unsigned StackAlign = getTypeStackAlignInBytes(Ty, TypeAlign);
  if (StackAlign == 0)
    return ABIArgInfo::getIndirect(4);

  // If the stack alignment is less than the type alignment, realign the
  // argument.
  if (StackAlign < TypeAlign)
    return ABIArgInfo::getIndirect(StackAlign, /*ByVal=*/true,
                                   /*Realign=*/true);

  return ABIArgInfo::getIndirect(StackAlign);
}

X86_32ABIInfo::Class X86_32ABIInfo::classify(QualType Ty) const {
  const Type *T = isSingleElementStruct(Ty, getContext());
  if (!T)
    T = Ty.getTypePtr();

  if (const BuiltinType *BT = T->getAs<BuiltinType>()) {
    BuiltinType::Kind K = BT->getKind();
    if (K == BuiltinType::Float || K == BuiltinType::Double)
      return Float;
  }
  return Integer;
}

bool X86_32ABIInfo::shouldUseInReg(QualType Ty, unsigned &FreeRegs,
                                   bool IsFastCall, bool &NeedsPadding) const {
  NeedsPadding = false;
  Class C = classify(Ty);
  if (C == Float)
    return false;

  unsigned Size = getContext().getTypeSize(Ty);
  unsigned SizeInRegs = (Size + 31) / 32;

  if (SizeInRegs == 0)
    return false;

  if (SizeInRegs > FreeRegs) {
    FreeRegs = 0;
    return false;
  }

  FreeRegs -= SizeInRegs;

  if (IsFastCall) {
    if (Size > 32)
      return false;

    if (Ty->isIntegralOrEnumerationType())
      return true;

    if (Ty->isPointerType())
      return true;

    if (Ty->isReferenceType())
      return true;

    if (FreeRegs)
      NeedsPadding = true;

    return false;
  }

  return true;
}

ABIArgInfo X86_32ABIInfo::classifyArgumentType(QualType Ty,
                                               unsigned &FreeRegs,
                                               bool IsFastCall) const {
  // FIXME: Set alignment on indirect arguments.
  if (isAggregateTypeForABI(Ty)) {
    // Structures with flexible arrays are always indirect.
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      // Structures with either a non-trivial destructor or a non-trivial
      // copy constructor are always indirect.
      if (hasNonTrivialDestructorOrCopyConstructor(RT))
        return getIndirectResult(Ty, false, FreeRegs);

      if (RT->getDecl()->hasFlexibleArrayMember())
        return getIndirectResult(Ty, true, FreeRegs);
    }

    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();

    llvm::LLVMContext &LLVMContext = getVMContext();
    llvm::IntegerType *Int32 = llvm::Type::getInt32Ty(LLVMContext);
    bool NeedsPadding;
    if (shouldUseInReg(Ty, FreeRegs, IsFastCall, NeedsPadding)) {
      unsigned SizeInRegs = (getContext().getTypeSize(Ty) + 31) / 32;
      SmallVector<llvm::Type*, 3> Elements;
      for (unsigned I = 0; I < SizeInRegs; ++I)
        Elements.push_back(Int32);
      llvm::Type *Result = llvm::StructType::get(LLVMContext, Elements);
      return ABIArgInfo::getDirectInReg(Result);
    }
    llvm::IntegerType *PaddingType = NeedsPadding ? Int32 : 0;

    // Expand small (<= 128-bit) record types when we know that the stack layout
    // of those arguments will match the struct. This is important because the
    // LLVM backend isn't smart enough to remove byval, which inhibits many
    // optimizations.
    if (getContext().getTypeSize(Ty) <= 4*32 &&
        canExpandIndirectArgument(Ty, getContext()))
      return ABIArgInfo::getExpandWithPadding(IsFastCall, PaddingType);

    return getIndirectResult(Ty, true, FreeRegs);
  }

  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // On Darwin, some vectors are passed in memory, we handle this by passing
    // it as an i8/i16/i32/i64.
    if (IsDarwinVectorABI) {
      uint64_t Size = getContext().getTypeSize(Ty);
      if ((Size == 8 || Size == 16 || Size == 32) ||
          (Size == 64 && VT->getNumElements() == 1))
        return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                            Size));
    }

    llvm::Type *IRType = CGT.ConvertType(Ty);
    if (UseX86_MMXType(IRType)) {
      if (IsMMXDisabled)
        return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                            64));
      ABIArgInfo AAI = ABIArgInfo::getDirect(IRType);
      AAI.setCoerceToType(llvm::Type::getX86_MMXTy(getVMContext()));
      return AAI;
    }

    return ABIArgInfo::getDirect();
  }


  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  bool NeedsPadding;
  bool InReg = shouldUseInReg(Ty, FreeRegs, IsFastCall, NeedsPadding);

  if (Ty->isPromotableIntegerType()) {
    if (InReg)
      return ABIArgInfo::getExtendInReg();
    return ABIArgInfo::getExtend();
  }
  if (InReg)
    return ABIArgInfo::getDirectInReg();
  return ABIArgInfo::getDirect();
}

void X86_32ABIInfo::computeInfo(CGFunctionInfo &FI) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType(),
                                          FI.getCallingConvention());

  unsigned CC = FI.getCallingConvention();
  bool IsFastCall = CC == llvm::CallingConv::X86_FastCall;
  unsigned FreeRegs;
  if (IsFastCall)
    FreeRegs = 2;
  else if (FI.getHasRegParm())
    FreeRegs = FI.getRegParm();
  else
    FreeRegs = DefaultNumRegisterParameters;

  // If the return value is indirect, then the hidden argument is consuming one
  // integer register.
  if (FI.getReturnInfo().isIndirect() && FreeRegs) {
    --FreeRegs;
    ABIArgInfo &Old = FI.getReturnInfo();
    Old = ABIArgInfo::getIndirectInReg(Old.getIndirectAlign(),
                                       Old.getIndirectByVal(),
                                       Old.getIndirectRealign());
  }

  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classifyArgumentType(it->type, FreeRegs, IsFastCall);
}

llvm::Value *X86_32ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");

  // Compute if the address needs to be aligned
  unsigned Align = CGF.getContext().getTypeAlignInChars(Ty).getQuantity();
  Align = getTypeStackAlignInBytes(Ty, Align);
  Align = std::max(Align, 4U);
  if (Align > 4) {
    // addr = (addr + align - 1) & -align;
    llvm::Value *Offset =
      llvm::ConstantInt::get(CGF.Int32Ty, Align - 1);
    Addr = CGF.Builder.CreateGEP(Addr, Offset);
    llvm::Value *AsInt = CGF.Builder.CreatePtrToInt(Addr,
                                                    CGF.Int32Ty);
    llvm::Value *Mask = llvm::ConstantInt::get(CGF.Int32Ty, -Align);
    Addr = CGF.Builder.CreateIntToPtr(CGF.Builder.CreateAnd(AsInt, Mask),
                                      Addr->getType(),
                                      "ap.cur.aligned");
  }

  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, Align);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
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
      llvm::AttrBuilder B;
      B.addStackAlignmentAttr(16);
      Fn->addAttribute(llvm::AttrListPtr::FunctionIndex,
                       llvm::Attributes::get(CGM.getLLVMContext(), B));
    }
  }
}

bool X86_32TargetCodeGenInfo::initDwarfEHRegSizeTable(
                                               CodeGen::CodeGenFunction &CGF,
                                               llvm::Value *Address) const {
  CodeGen::CGBuilderTy &Builder = CGF.Builder;

  llvm::Value *Four8 = llvm::ConstantInt::get(CGF.Int8Ty, 4);

  // 0-7 are the eight integer registers;  the order is different
  //   on Darwin (for EH), but the range is the same.
  // 8 is %eip.
  AssignToArrayRange(Builder, Address, Four8, 0, 8);

  if (CGF.CGM.isTargetDarwin()) {
    // 12-16 are st(0..4).  Not sure why we stop at 4.
    // These have size 16, which is sizeof(long double) on
    // platforms with 8-byte alignment for that type.
    llvm::Value *Sixteen8 = llvm::ConstantInt::get(CGF.Int8Ty, 16);
    AssignToArrayRange(Builder, Address, Sixteen8, 12, 16);

  } else {
    // 9 is %eflags, which doesn't get a size on Darwin for some
    // reason.
    Builder.CreateStore(Four8, Builder.CreateConstInBoundsGEP1_32(Address, 9));

    // 11-16 are st(0..5).  Not sure why we stop at 5.
    // These have size 12, which is sizeof(long double) on
    // platforms with 4-byte alignment for that type.
    llvm::Value *Twelve8 = llvm::ConstantInt::get(CGF.Int8Ty, 12);
    AssignToArrayRange(Builder, Address, Twelve8, 11, 16);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// X86-64 ABI Implementation
//===----------------------------------------------------------------------===//


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
  static Class merge(Class Accum, Class Field);

  /// postMerge - Implement the X86_64 ABI post merging algorithm.
  ///
  /// Post merger cleanup, reduces a malformed Hi and Lo pair to
  /// final MEMORY or SSE classes when necessary.
  ///
  /// \param AggregateSize - The size of the current aggregate in
  /// the classification process.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the higher words of the containing object.
  ///
  void postMerge(unsigned AggregateSize, Class &Lo, Class &Hi) const;

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
  void classify(QualType T, uint64_t OffsetBase, Class &Lo, Class &Hi) const;

  llvm::Type *GetByteVectorType(QualType Ty) const;
  llvm::Type *GetSSETypeAtOffset(llvm::Type *IRType,
                                 unsigned IROffset, QualType SourceTy,
                                 unsigned SourceOffset) const;
  llvm::Type *GetINTEGERTypeAtOffset(llvm::Type *IRType,
                                     unsigned IROffset, QualType SourceTy,
                                     unsigned SourceOffset) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be returned in memory.
  ABIArgInfo getIndirectReturnResult(QualType Ty) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ///
  /// \param freeIntRegs - The number of free integer registers remaining
  /// available.
  ABIArgInfo getIndirectResult(QualType Ty, unsigned freeIntRegs) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;

  ABIArgInfo classifyArgumentType(QualType Ty,
                                  unsigned freeIntRegs,
                                  unsigned &neededInt,
                                  unsigned &neededSSE) const;

  bool IsIllegalVectorType(QualType Ty) const;

  /// The 0.98 ABI revision clarified a lot of ambiguities,
  /// unfortunately in ways that were not always consistent with
  /// certain previous compilers.  In particular, platforms which
  /// required strict binary compatibility with older versions of GCC
  /// may need to exempt themselves.
  bool honorsRevision0_98() const {
    return !getContext().getTargetInfo().getTriple().isOSDarwin();
  }

  bool HasAVX;
  // Some ABIs (e.g. X32 ABI and Native Client OS) use 32 bit pointers on
  // 64-bit hardware.
  bool Has64BitPointers;

public:
  X86_64ABIInfo(CodeGen::CodeGenTypes &CGT, bool hasavx) :
      ABIInfo(CGT), HasAVX(hasavx),
      Has64BitPointers(CGT.getDataLayout().getPointerSize(0) == 8) {
  }

  bool isPassedUsingAVXType(QualType type) const {
    unsigned neededInt, neededSSE;
    // The freeIntRegs argument doesn't matter here.
    ABIArgInfo info = classifyArgumentType(type, 0, neededInt, neededSSE);
    if (info.isDirect()) {
      llvm::Type *ty = info.getCoerceToType();
      if (llvm::VectorType *vectorTy = dyn_cast_or_null<llvm::VectorType>(ty))
        return (vectorTy->getBitWidth() > 128);
    }
    return false;
  }

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

/// WinX86_64ABIInfo - The Windows X86_64 ABI information.
class WinX86_64ABIInfo : public ABIInfo {

  ABIArgInfo classify(QualType Ty) const;

public:
  WinX86_64ABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class X86_64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, bool HasAVX)
      : TargetCodeGenInfo(new X86_64ABIInfo(CGT, HasAVX)) {}

  const X86_64ABIInfo &getABIInfo() const {
    return static_cast<const X86_64ABIInfo&>(TargetCodeGenInfo::getABIInfo());
  }

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    llvm::Value *Eight8 = llvm::ConstantInt::get(CGF.Int8Ty, 8);

    // 0-15 are the 16 integer registers.
    // 16 is %rip.
    AssignToArrayRange(CGF.Builder, Address, Eight8, 0, 16);
    return false;
  }

  llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                  StringRef Constraint,
                                  llvm::Type* Ty) const {
    return X86AdjustInlineAsmType(CGF, Constraint, Ty);
  }

  bool isNoProtoCallVariadic(const CallArgList &args,
                             const FunctionNoProtoType *fnType) const {
    // The default CC on x86-64 sets %al to the number of SSA
    // registers used, and GCC sets this when calling an unprototyped
    // function, so we override the default behavior.  However, don't do
    // that when AVX types are involved: the ABI explicitly states it is
    // undefined, and it doesn't work in practice because of how the ABI
    // defines varargs anyway.
    if (fnType->getCallConv() == CC_Default || fnType->getCallConv() == CC_C) {
      bool HasAVXType = false;
      for (CallArgList::const_iterator
             it = args.begin(), ie = args.end(); it != ie; ++it) {
        if (getABIInfo().isPassedUsingAVXType(it->Ty)) {
          HasAVXType = true;
          break;
        }
      }

      if (!HasAVXType)
        return true;
    }

    return TargetCodeGenInfo::isNoProtoCallVariadic(args, fnType);
  }

};

class WinX86_64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  WinX86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new WinX86_64ABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    llvm::Value *Eight8 = llvm::ConstantInt::get(CGF.Int8Ty, 8);

    // 0-15 are the 16 integer registers.
    // 16 is %rip.
    AssignToArrayRange(CGF.Builder, Address, Eight8, 0, 16);
    return false;
  }
};

}

void X86_64ABIInfo::postMerge(unsigned AggregateSize, Class &Lo,
                              Class &Hi) const {
  // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
  //
  // (a) If one of the classes is Memory, the whole argument is passed in
  //     memory.
  //
  // (b) If X87UP is not preceded by X87, the whole argument is passed in
  //     memory.
  //
  // (c) If the size of the aggregate exceeds two eightbytes and the first
  //     eightbyte isn't SSE or any other eightbyte isn't SSEUP, the whole
  //     argument is passed in memory. NOTE: This is necessary to keep the
  //     ABI working for processors that don't support the __m256 type.
  //
  // (d) If SSEUP is not preceded by SSE or SSEUP, it is converted to SSE.
  //
  // Some of these are enforced by the merging logic.  Others can arise
  // only with unions; for example:
  //   union { _Complex double; unsigned; }
  //
  // Note that clauses (b) and (c) were added in 0.98.
  //
  if (Hi == Memory)
    Lo = Memory;
  if (Hi == X87Up && Lo != X87 && honorsRevision0_98())
    Lo = Memory;
  if (AggregateSize > 128 && (Lo != SSE || Hi != SSEUp))
    Lo = Memory;
  if (Hi == SSEUp && Lo != SSE)
    Hi = SSE;
}

X86_64ABIInfo::Class X86_64ABIInfo::merge(Class Accum, Class Field) {
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
  if (Field == Memory)
    return Memory;
  if (Accum == NoClass)
    return Field;
  if (Accum == Integer || Field == Integer)
    return Integer;
  if (Field == X87 || Field == X87Up || Field == ComplexX87 ||
      Accum == X87 || Accum == X87Up)
    return Memory;
  return SSE;
}

void X86_64ABIInfo::classify(QualType Ty, uint64_t OffsetBase,
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
    } else if ((k == BuiltinType::Float || k == BuiltinType::Double) ||
               (k == BuiltinType::LongDouble &&
                getContext().getTargetInfo().getTriple().getOS() ==
                llvm::Triple::NativeClient)) {
      Current = SSE;
    } else if (k == BuiltinType::LongDouble) {
      Lo = X87;
      Hi = X87Up;
    }
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }

  if (const EnumType *ET = Ty->getAs<EnumType>()) {
    // Classify the underlying integer type.
    classify(ET->getDecl()->getIntegerType(), OffsetBase, Lo, Hi);
    return;
  }

  if (Ty->hasPointerRepresentation()) {
    Current = Integer;
    return;
  }

  if (Ty->isMemberPointerType()) {
    if (Ty->isMemberFunctionPointerType() && Has64BitPointers)
      Lo = Hi = Integer;
    else
      Current = Integer;
    return;
  }

  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    uint64_t Size = getContext().getTypeSize(VT);
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
      if (VT->getElementType()->isSpecificBuiltinType(BuiltinType::LongLong) ||
          VT->getElementType()->isSpecificBuiltinType(BuiltinType::ULongLong) ||
          VT->getElementType()->isSpecificBuiltinType(BuiltinType::Long) ||
          VT->getElementType()->isSpecificBuiltinType(BuiltinType::ULong))
        Current = Integer;
      else
        Current = SSE;

      // If this type crosses an eightbyte boundary, it should be
      // split.
      if (OffsetBase && OffsetBase != 64)
        Hi = Lo;
    } else if (Size == 128 || (HasAVX && Size == 256)) {
      // Arguments of 256-bits are split into four eightbyte chunks. The
      // least significant one belongs to class SSE and all the others to class
      // SSEUP. The original Lo and Hi design considers that types can't be
      // greater than 128-bits, so a 64-bit split in Hi and Lo makes sense.
      // This design isn't correct for 256-bits, but since there're no cases
      // where the upper parts would need to be inspected, avoid adding
      // complexity and just consider Hi to match the 64-256 part.
      Lo = SSE;
      Hi = SSEUp;
    }
    return;
  }

  if (const ComplexType *CT = Ty->getAs<ComplexType>()) {
    QualType ET = getContext().getCanonicalType(CT->getElementType());

    uint64_t Size = getContext().getTypeSize(Ty);
    if (ET->isIntegralOrEnumerationType()) {
      if (Size <= 64)
        Current = Integer;
      else if (Size <= 128)
        Lo = Hi = Integer;
    } else if (ET == getContext().FloatTy)
      Current = SSE;
    else if (ET == getContext().DoubleTy ||
             (ET == getContext().LongDoubleTy &&
              getContext().getTargetInfo().getTriple().getOS() ==
              llvm::Triple::NativeClient))
      Lo = Hi = SSE;
    else if (ET == getContext().LongDoubleTy)
      Current = ComplexX87;

    // If this complex type crosses an eightbyte boundary then it
    // should be split.
    uint64_t EB_Real = (OffsetBase) / 64;
    uint64_t EB_Imag = (OffsetBase + getContext().getTypeSize(ET)) / 64;
    if (Hi == NoClass && EB_Real != EB_Imag)
      Hi = Lo;

    return;
  }

  if (const ConstantArrayType *AT = getContext().getAsConstantArrayType(Ty)) {
    // Arrays are treated like structures.

    uint64_t Size = getContext().getTypeSize(Ty);

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than four eightbytes, ..., it has class MEMORY.
    if (Size > 256)
      return;

    // AMD64-ABI 3.2.3p2: Rule 1. If ..., or it contains unaligned
    // fields, it has class MEMORY.
    //
    // Only need to check alignment of array base.
    if (OffsetBase % getContext().getTypeAlign(AT->getElementType()))
      return;

    // Otherwise implement simplified merge. We could be smarter about
    // this, but it isn't worth it and would be harder to verify.
    Current = NoClass;
    uint64_t EltSize = getContext().getTypeSize(AT->getElementType());
    uint64_t ArraySize = AT->getSize().getZExtValue();

    // The only case a 256-bit wide vector could be used is when the array
    // contains a single 256-bit element. Since Lo and Hi logic isn't extended
    // to work for sizes wider than 128, early check and fallback to memory.
    if (Size > 128 && EltSize != 256)
      return;

    for (uint64_t i=0, Offset=OffsetBase; i<ArraySize; ++i, Offset += EltSize) {
      Class FieldLo, FieldHi;
      classify(AT->getElementType(), Offset, FieldLo, FieldHi);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }

    postMerge(Size, Lo, Hi);
    assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp array classification.");
    return;
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    uint64_t Size = getContext().getTypeSize(Ty);

    // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
    // than four eightbytes, ..., it has class MEMORY.
    if (Size > 256)
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

    const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

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
        uint64_t Offset =
          OffsetBase + getContext().toBits(Layout.getBaseClassOffset(Base));
        classify(i->getType(), Offset, FieldLo, FieldHi);
        Lo = merge(Lo, FieldLo);
        Hi = merge(Hi, FieldHi);
        if (Lo == Memory || Hi == Memory)
          break;
      }
    }

    // Classify the fields one at a time, merging the results.
    unsigned idx = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
           i != e; ++i, ++idx) {
      uint64_t Offset = OffsetBase + Layout.getFieldOffset(idx);
      bool BitField = i->isBitField();

      // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger than
      // four eightbytes, or it contains unaligned fields, it has class MEMORY.
      //
      // The only case a 256-bit wide vector could be used is when the struct
      // contains a single 256-bit element. Since Lo and Hi logic isn't extended
      // to work for sizes wider than 128, early check and fallback to memory.
      //
      if (Size > 128 && getContext().getTypeSize(i->getType()) != 256) {
        Lo = Memory;
        return;
      }
      // Note, skip this test for bit-fields, see below.
      if (!BitField && Offset % getContext().getTypeAlign(i->getType())) {
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
        uint64_t Size = i->getBitWidthValue(getContext());

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
        classify(i->getType(), Offset, FieldLo, FieldHi);
      Lo = merge(Lo, FieldLo);
      Hi = merge(Hi, FieldHi);
      if (Lo == Memory || Hi == Memory)
        break;
    }

    postMerge(Size, Lo, Hi);
  }
}

ABIArgInfo X86_64ABIInfo::getIndirectReturnResult(QualType Ty) const {
  // If this is a scalar LLVM value then assume LLVM will pass it in the right
  // place naturally.
  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  return ABIArgInfo::getIndirect(0);
}

bool X86_64ABIInfo::IsIllegalVectorType(QualType Ty) const {
  if (const VectorType *VecTy = Ty->getAs<VectorType>()) {
    uint64_t Size = getContext().getTypeSize(VecTy);
    unsigned LargestVector = HasAVX ? 256 : 128;
    if (Size <= 64 || Size > LargestVector)
      return true;
  }

  return false;
}

ABIArgInfo X86_64ABIInfo::getIndirectResult(QualType Ty,
                                            unsigned freeIntRegs) const {
  // If this is a scalar LLVM value then assume LLVM will pass it in the right
  // place naturally.
  //
  // This assumption is optimistic, as there could be free registers available
  // when we need to pass this argument in memory, and LLVM could try to pass
  // the argument in the free register. This does not seem to happen currently,
  // but this code would be much safer if we could mark the argument with
  // 'onstack'. See PR12193.
  if (!isAggregateTypeForABI(Ty) && !IsIllegalVectorType(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Compute the byval alignment. We specify the alignment of the byval in all
  // cases so that the mid-level optimizer knows the alignment of the byval.
  unsigned Align = std::max(getContext().getTypeAlign(Ty) / 8, 8U);

  // Attempt to avoid passing indirect results using byval when possible. This
  // is important for good codegen.
  //
  // We do this by coercing the value into a scalar type which the backend can
  // handle naturally (i.e., without using byval).
  //
  // For simplicity, we currently only do this when we have exhausted all of the
  // free integer registers. Doing this when there are free integer registers
  // would require more care, as we would have to ensure that the coerced value
  // did not claim the unused register. That would require either reording the
  // arguments to the function (so that any subsequent inreg values came first),
  // or only doing this optimization when there were no following arguments that
  // might be inreg.
  //
  // We currently expect it to be rare (particularly in well written code) for
  // arguments to be passed on the stack when there are still free integer
  // registers available (this would typically imply large structs being passed
  // by value), so this seems like a fair tradeoff for now.
  //
  // We can revisit this if the backend grows support for 'onstack' parameter
  // attributes. See PR12193.
  if (freeIntRegs == 0) {
    uint64_t Size = getContext().getTypeSize(Ty);

    // If this type fits in an eightbyte, coerce it into the matching integral
    // type, which will end up on the stack (with alignment 8).
    if (Align == 8 && Size <= 64)
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                          Size));
  }

  return ABIArgInfo::getIndirect(Align);
}

/// GetByteVectorType - The ABI specifies that a value should be passed in an
/// full vector XMM/YMM register.  Pick an LLVM IR type that will be passed as a
/// vector register.
llvm::Type *X86_64ABIInfo::GetByteVectorType(QualType Ty) const {
  llvm::Type *IRType = CGT.ConvertType(Ty);

  // Wrapper structs that just contain vectors are passed just like vectors,
  // strip them off if present.
  llvm::StructType *STy = dyn_cast<llvm::StructType>(IRType);
  while (STy && STy->getNumElements() == 1) {
    IRType = STy->getElementType(0);
    STy = dyn_cast<llvm::StructType>(IRType);
  }

  // If the preferred type is a 16-byte vector, prefer to pass it.
  if (llvm::VectorType *VT = dyn_cast<llvm::VectorType>(IRType)){
    llvm::Type *EltTy = VT->getElementType();
    unsigned BitWidth = VT->getBitWidth();
    if ((BitWidth >= 128 && BitWidth <= 256) &&
        (EltTy->isFloatTy() || EltTy->isDoubleTy() ||
         EltTy->isIntegerTy(8) || EltTy->isIntegerTy(16) ||
         EltTy->isIntegerTy(32) || EltTy->isIntegerTy(64) ||
         EltTy->isIntegerTy(128)))
      return VT;
  }

  return llvm::VectorType::get(llvm::Type::getDoubleTy(getVMContext()), 2);
}

/// BitsContainNoUserData - Return true if the specified [start,end) bit range
/// is known to either be off the end of the specified type or being in
/// alignment padding.  The user type specified is known to be at most 128 bits
/// in size, and have passed through X86_64ABIInfo::classify with a successful
/// classification that put one of the two halves in the INTEGER class.
///
/// It is conservatively correct to return false.
static bool BitsContainNoUserData(QualType Ty, unsigned StartBit,
                                  unsigned EndBit, ASTContext &Context) {
  // If the bytes being queried are off the end of the type, there is no user
  // data hiding here.  This handles analysis of builtins, vectors and other
  // types that don't contain interesting padding.
  unsigned TySize = (unsigned)Context.getTypeSize(Ty);
  if (TySize <= StartBit)
    return true;

  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty)) {
    unsigned EltSize = (unsigned)Context.getTypeSize(AT->getElementType());
    unsigned NumElts = (unsigned)AT->getSize().getZExtValue();

    // Check each element to see if the element overlaps with the queried range.
    for (unsigned i = 0; i != NumElts; ++i) {
      // If the element is after the span we care about, then we're done..
      unsigned EltOffset = i*EltSize;
      if (EltOffset >= EndBit) break;

      unsigned EltStart = EltOffset < StartBit ? StartBit-EltOffset :0;
      if (!BitsContainNoUserData(AT->getElementType(), EltStart,
                                 EndBit-EltOffset, Context))
        return false;
    }
    // If it overlaps no elements, then it is safe to process as padding.
    return true;
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    // If this is a C++ record, check the bases first.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (CXXRecordDecl::base_class_const_iterator i = CXXRD->bases_begin(),
           e = CXXRD->bases_end(); i != e; ++i) {
        assert(!i->isVirtual() && !i->getType()->isDependentType() &&
               "Unexpected base class!");
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

        // If the base is after the span we care about, ignore it.
        unsigned BaseOffset = Context.toBits(Layout.getBaseClassOffset(Base));
        if (BaseOffset >= EndBit) continue;

        unsigned BaseStart = BaseOffset < StartBit ? StartBit-BaseOffset :0;
        if (!BitsContainNoUserData(i->getType(), BaseStart,
                                   EndBit-BaseOffset, Context))
          return false;
      }
    }

    // Verify that no field has data that overlaps the region of interest.  Yes
    // this could be sped up a lot by being smarter about queried fields,
    // however we're only looking at structs up to 16 bytes, so we don't care
    // much.
    unsigned idx = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i, ++idx) {
      unsigned FieldOffset = (unsigned)Layout.getFieldOffset(idx);

      // If we found a field after the region we care about, then we're done.
      if (FieldOffset >= EndBit) break;

      unsigned FieldStart = FieldOffset < StartBit ? StartBit-FieldOffset :0;
      if (!BitsContainNoUserData(i->getType(), FieldStart, EndBit-FieldOffset,
                                 Context))
        return false;
    }

    // If nothing in this record overlapped the area of interest, then we're
    // clean.
    return true;
  }

  return false;
}

/// ContainsFloatAtOffset - Return true if the specified LLVM IR type has a
/// float member at the specified offset.  For example, {int,{float}} has a
/// float at offset 4.  It is conservatively correct for this routine to return
/// false.
static bool ContainsFloatAtOffset(llvm::Type *IRType, unsigned IROffset,
                                  const llvm::DataLayout &TD) {
  // Base case if we find a float.
  if (IROffset == 0 && IRType->isFloatTy())
    return true;

  // If this is a struct, recurse into the field at the specified offset.
  if (llvm::StructType *STy = dyn_cast<llvm::StructType>(IRType)) {
    const llvm::StructLayout *SL = TD.getStructLayout(STy);
    unsigned Elt = SL->getElementContainingOffset(IROffset);
    IROffset -= SL->getElementOffset(Elt);
    return ContainsFloatAtOffset(STy->getElementType(Elt), IROffset, TD);
  }

  // If this is an array, recurse into the field at the specified offset.
  if (llvm::ArrayType *ATy = dyn_cast<llvm::ArrayType>(IRType)) {
    llvm::Type *EltTy = ATy->getElementType();
    unsigned EltSize = TD.getTypeAllocSize(EltTy);
    IROffset -= IROffset/EltSize*EltSize;
    return ContainsFloatAtOffset(EltTy, IROffset, TD);
  }

  return false;
}


/// GetSSETypeAtOffset - Return a type that will be passed by the backend in the
/// low 8 bytes of an XMM register, corresponding to the SSE class.
llvm::Type *X86_64ABIInfo::
GetSSETypeAtOffset(llvm::Type *IRType, unsigned IROffset,
                   QualType SourceTy, unsigned SourceOffset) const {
  // The only three choices we have are either double, <2 x float>, or float. We
  // pass as float if the last 4 bytes is just padding.  This happens for
  // structs that contain 3 floats.
  if (BitsContainNoUserData(SourceTy, SourceOffset*8+32,
                            SourceOffset*8+64, getContext()))
    return llvm::Type::getFloatTy(getVMContext());

  // We want to pass as <2 x float> if the LLVM IR type contains a float at
  // offset+0 and offset+4.  Walk the LLVM IR type to find out if this is the
  // case.
  if (ContainsFloatAtOffset(IRType, IROffset, getDataLayout()) &&
      ContainsFloatAtOffset(IRType, IROffset+4, getDataLayout()))
    return llvm::VectorType::get(llvm::Type::getFloatTy(getVMContext()), 2);

  return llvm::Type::getDoubleTy(getVMContext());
}


/// GetINTEGERTypeAtOffset - The ABI specifies that a value should be passed in
/// an 8-byte GPR.  This means that we either have a scalar or we are talking
/// about the high or low part of an up-to-16-byte struct.  This routine picks
/// the best LLVM IR type to represent this, which may be i64 or may be anything
/// else that the backend will pass in a GPR that works better (e.g. i8, %foo*,
/// etc).
///
/// PrefType is an LLVM IR type that corresponds to (part of) the IR type for
/// the source type.  IROffset is an offset in bytes into the LLVM IR type that
/// the 8-byte value references.  PrefType may be null.
///
/// SourceTy is the source level type for the entire argument.  SourceOffset is
/// an offset into this that we're processing (which is always either 0 or 8).
///
llvm::Type *X86_64ABIInfo::
GetINTEGERTypeAtOffset(llvm::Type *IRType, unsigned IROffset,
                       QualType SourceTy, unsigned SourceOffset) const {
  // If we're dealing with an un-offset LLVM IR type, then it means that we're
  // returning an 8-byte unit starting with it.  See if we can safely use it.
  if (IROffset == 0) {
    // Pointers and int64's always fill the 8-byte unit.
    if ((isa<llvm::PointerType>(IRType) && Has64BitPointers) ||
        IRType->isIntegerTy(64))
      return IRType;

    // If we have a 1/2/4-byte integer, we can use it only if the rest of the
    // goodness in the source type is just tail padding.  This is allowed to
    // kick in for struct {double,int} on the int, but not on
    // struct{double,int,int} because we wouldn't return the second int.  We
    // have to do this analysis on the source type because we can't depend on
    // unions being lowered a specific way etc.
    if (IRType->isIntegerTy(8) || IRType->isIntegerTy(16) ||
        IRType->isIntegerTy(32) ||
        (isa<llvm::PointerType>(IRType) && !Has64BitPointers)) {
      unsigned BitWidth = isa<llvm::PointerType>(IRType) ? 32 :
          cast<llvm::IntegerType>(IRType)->getBitWidth();

      if (BitsContainNoUserData(SourceTy, SourceOffset*8+BitWidth,
                                SourceOffset*8+64, getContext()))
        return IRType;
    }
  }

  if (llvm::StructType *STy = dyn_cast<llvm::StructType>(IRType)) {
    // If this is a struct, recurse into the field at the specified offset.
    const llvm::StructLayout *SL = getDataLayout().getStructLayout(STy);
    if (IROffset < SL->getSizeInBytes()) {
      unsigned FieldIdx = SL->getElementContainingOffset(IROffset);
      IROffset -= SL->getElementOffset(FieldIdx);

      return GetINTEGERTypeAtOffset(STy->getElementType(FieldIdx), IROffset,
                                    SourceTy, SourceOffset);
    }
  }

  if (llvm::ArrayType *ATy = dyn_cast<llvm::ArrayType>(IRType)) {
    llvm::Type *EltTy = ATy->getElementType();
    unsigned EltSize = getDataLayout().getTypeAllocSize(EltTy);
    unsigned EltOffset = IROffset/EltSize*EltSize;
    return GetINTEGERTypeAtOffset(EltTy, IROffset-EltOffset, SourceTy,
                                  SourceOffset);
  }

  // Okay, we don't have any better idea of what to pass, so we pass this in an
  // integer register that isn't too big to fit the rest of the struct.
  unsigned TySizeInBytes =
    (unsigned)getContext().getTypeSizeInChars(SourceTy).getQuantity();

  assert(TySizeInBytes != SourceOffset && "Empty field?");

  // It is always safe to classify this as an integer type up to i64 that
  // isn't larger than the structure.
  return llvm::IntegerType::get(getVMContext(),
                                std::min(TySizeInBytes-SourceOffset, 8U)*8);
}


/// GetX86_64ByValArgumentPair - Given a high and low type that can ideally
/// be used as elements of a two register pair to pass or return, return a
/// first class aggregate to represent them.  For example, if the low part of
/// a by-value argument should be passed as i32* and the high part as float,
/// return {i32*, float}.
static llvm::Type *
GetX86_64ByValArgumentPair(llvm::Type *Lo, llvm::Type *Hi,
                           const llvm::DataLayout &TD) {
  // In order to correctly satisfy the ABI, we need to the high part to start
  // at offset 8.  If the high and low parts we inferred are both 4-byte types
  // (e.g. i32 and i32) then the resultant struct type ({i32,i32}) won't have
  // the second element at offset 8.  Check for this:
  unsigned LoSize = (unsigned)TD.getTypeAllocSize(Lo);
  unsigned HiAlign = TD.getABITypeAlignment(Hi);
  unsigned HiStart = llvm::DataLayout::RoundUpAlignment(LoSize, HiAlign);
  assert(HiStart != 0 && HiStart <= 8 && "Invalid x86-64 argument pair!");

  // To handle this, we have to increase the size of the low part so that the
  // second element will start at an 8 byte offset.  We can't increase the size
  // of the second element because it might make us access off the end of the
  // struct.
  if (HiStart != 8) {
    // There are only two sorts of types the ABI generation code can produce for
    // the low part of a pair that aren't 8 bytes in size: float or i8/i16/i32.
    // Promote these to a larger type.
    if (Lo->isFloatTy())
      Lo = llvm::Type::getDoubleTy(Lo->getContext());
    else {
      assert(Lo->isIntegerTy() && "Invalid/unknown lo type");
      Lo = llvm::Type::getInt64Ty(Lo->getContext());
    }
  }

  llvm::StructType *Result = llvm::StructType::get(Lo, Hi, NULL);


  // Verify that the second element is at an 8-byte offset.
  assert(TD.getStructLayout(Result)->getElementOffset(1) == 8 &&
         "Invalid x86-64 argument pair!");
  return Result;
}

ABIArgInfo X86_64ABIInfo::
classifyReturnType(QualType RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi);

  // Check some invariants.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  llvm::Type *ResType = 0;
  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ABIArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == SSE || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;

  case SSEUp:
  case X87Up:
    llvm_unreachable("Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p4: Rule 2. Types of class memory are returned via
    // hidden argument.
  case Memory:
    return getIndirectReturnResult(RetTy);

    // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next
    // available register of the sequence %rax, %rdx is used.
  case Integer:
    ResType = GetINTEGERTypeAtOffset(CGT.ConvertType(RetTy), 0, RetTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && isa<llvm::IntegerType>(ResType)) {
      // Treat an enum type as its underlying type.
      if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
        RetTy = EnumTy->getDecl()->getIntegerType();

      if (RetTy->isIntegralOrEnumerationType() &&
          RetTy->isPromotableIntegerType())
        return ABIArgInfo::getExtend();
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case SSE:
    ResType = GetSSETypeAtOffset(CGT.ConvertType(RetTy), 0, RetTy, 0);
    break;

    // AMD64-ABI 3.2.3p4: Rule 6. If the class is X87, the value is
    // returned on the X87 stack in %st0 as 80-bit x87 number.
  case X87:
    ResType = llvm::Type::getX86_FP80Ty(getVMContext());
    break;

    // AMD64-ABI 3.2.3p4: Rule 8. If the class is COMPLEX_X87, the real
    // part of the value is returned in %st0 and the imaginary part in
    // %st1.
  case ComplexX87:
    assert(Hi == ComplexX87 && "Unexpected ComplexX87 classification.");
    ResType = llvm::StructType::get(llvm::Type::getX86_FP80Ty(getVMContext()),
                                    llvm::Type::getX86_FP80Ty(getVMContext()),
                                    NULL);
    break;
  }

  llvm::Type *HighPart = 0;
  switch (Hi) {
    // Memory was handled previously and X87 should
    // never occur as a hi class.
  case Memory:
  case X87:
    llvm_unreachable("Invalid classification for hi word.");

  case ComplexX87: // Previously handled.
  case NoClass:
    break;

  case Integer:
    HighPart = GetINTEGERTypeAtOffset(CGT.ConvertType(RetTy), 8, RetTy, 8);
    if (Lo == NoClass)  // Return HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;
  case SSE:
    HighPart = GetSSETypeAtOffset(CGT.ConvertType(RetTy), 8, RetTy, 8);
    if (Lo == NoClass)  // Return HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

    // AMD64-ABI 3.2.3p4: Rule 5. If the class is SSEUP, the eightbyte
    // is passed in the next available eightbyte chunk if the last used
    // vector register.
    //
    // SSEUP should always be preceded by SSE, just widen.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification.");
    ResType = GetByteVectorType(RetTy);
    break;

    // AMD64-ABI 3.2.3p4: Rule 7. If the class is X87UP, the value is
    // returned together with the previous X87 value in %st0.
  case X87Up:
    // If X87Up is preceded by X87, we don't need to do
    // anything. However, in some cases with unions it may not be
    // preceded by X87. In such situations we follow gcc and pass the
    // extra bits in an SSE reg.
    if (Lo != X87) {
      HighPart = GetSSETypeAtOffset(CGT.ConvertType(RetTy), 8, RetTy, 8);
      if (Lo == NoClass)  // Return HighPart at offset 8 in memory.
        return ABIArgInfo::getDirect(HighPart, 8);
    }
    break;
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = GetX86_64ByValArgumentPair(ResType, HighPart, getDataLayout());

  return ABIArgInfo::getDirect(ResType);
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(
  QualType Ty, unsigned freeIntRegs, unsigned &neededInt, unsigned &neededSSE)
  const
{
  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi);

  // Check some invariants.
  // FIXME: Enforce these by construction.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  llvm::Type *ResType = 0;
  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ABIArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == SSE || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;

    // AMD64-ABI 3.2.3p3: Rule 1. If the class is MEMORY, pass the argument
    // on the stack.
  case Memory:

    // AMD64-ABI 3.2.3p3: Rule 5. If the class is X87, X87UP or
    // COMPLEX_X87, it is passed in memory.
  case X87:
  case ComplexX87:
    if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
      ++neededInt;
    return getIndirectResult(Ty, freeIntRegs);

  case SSEUp:
  case X87Up:
    llvm_unreachable("Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next
    // available register of the sequence %rdi, %rsi, %rdx, %rcx, %r8
    // and %r9 is used.
  case Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    ResType = GetINTEGERTypeAtOffset(CGT.ConvertType(Ty), 0, Ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (Hi == NoClass && isa<llvm::IntegerType>(ResType)) {
      // Treat an enum type as its underlying type.
      if (const EnumType *EnumTy = Ty->getAs<EnumType>())
        Ty = EnumTy->getDecl()->getIntegerType();

      if (Ty->isIntegralOrEnumerationType() &&
          Ty->isPromotableIntegerType())
        return ABIArgInfo::getExtend();
    }

    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next
    // available SSE register is used, the registers are taken in the
    // order from %xmm0 to %xmm7.
  case SSE: {
    llvm::Type *IRType = CGT.ConvertType(Ty);
    ResType = GetSSETypeAtOffset(IRType, 0, Ty, 0);
    ++neededSSE;
    break;
  }
  }

  llvm::Type *HighPart = 0;
  switch (Hi) {
    // Memory was handled previously, ComplexX87 and X87 should
    // never occur as hi classes, and X87Up must be preceded by X87,
    // which is passed in memory.
  case Memory:
  case X87:
  case ComplexX87:
    llvm_unreachable("Invalid classification for hi word.");

  case NoClass: break;

  case Integer:
    ++neededInt;
    // Pick an 8-byte type based on the preferred type.
    HighPart = GetINTEGERTypeAtOffset(CGT.ConvertType(Ty), 8, Ty, 8);

    if (Lo == NoClass)  // Pass HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

    // X87Up generally doesn't occur here (long double is passed in
    // memory), except in situations involving unions.
  case X87Up:
  case SSE:
    HighPart = GetSSETypeAtOffset(CGT.ConvertType(Ty), 8, Ty, 8);

    if (Lo == NoClass)  // Pass HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);

    ++neededSSE;
    break;

    // AMD64-ABI 3.2.3p3: Rule 4. If the class is SSEUP, the
    // eightbyte is passed in the upper half of the last used SSE
    // register.  This only happens when 128-bit vectors are passed.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification");
    ResType = GetByteVectorType(Ty);
    break;
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming a
  // first class struct aggregate with the high and low part: {low, high}
  if (HighPart)
    ResType = GetX86_64ByValArgumentPair(ResType, HighPart, getDataLayout());

  return ABIArgInfo::getDirect(ResType);
}

void X86_64ABIInfo::computeInfo(CGFunctionInfo &FI) const {

  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

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
    it->info = classifyArgumentType(it->type, freeIntRegs, neededInt,
                                    neededSSE);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (freeIntRegs >= neededInt && freeSSERegs >= neededSSE) {
      freeIntRegs -= neededInt;
      freeSSERegs -= neededSSE;
    } else {
      it->info = getIndirectResult(it->type, freeIntRegs);
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
  // It isn't stated explicitly in the standard, but in practice we use
  // alignment greater than 16 where necessary.
  uint64_t Align = CGF.getContext().getTypeAlign(Ty) / 8;
  if (Align > 8) {
    // overflow_arg_area = (overflow_arg_area + align - 1) & -align;
    llvm::Value *Offset =
      llvm::ConstantInt::get(CGF.Int64Ty, Align - 1);
    overflow_arg_area = CGF.Builder.CreateGEP(overflow_arg_area, Offset);
    llvm::Value *AsInt = CGF.Builder.CreatePtrToInt(overflow_arg_area,
                                                    CGF.Int64Ty);
    llvm::Value *Mask = llvm::ConstantInt::get(CGF.Int64Ty, -(uint64_t)Align);
    overflow_arg_area =
      CGF.Builder.CreateIntToPtr(CGF.Builder.CreateAnd(AsInt, Mask),
                                 overflow_arg_area->getType(),
                                 "overflow_arg_area.align");
  }

  // AMD64-ABI 3.5.7p5: Step 8. Fetch type from l->overflow_arg_area.
  llvm::Type *LTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *Res =
    CGF.Builder.CreateBitCast(overflow_arg_area,
                              llvm::PointerType::getUnqual(LTy));

  // AMD64-ABI 3.5.7p5: Step 9. Set l->overflow_arg_area to:
  // l->overflow_arg_area + sizeof(type).
  // AMD64-ABI 3.5.7p5: Step 10. Align l->overflow_arg_area upwards to
  // an 8 byte boundary.

  uint64_t SizeInBytes = (CGF.getContext().getTypeSize(Ty) + 7) / 8;
  llvm::Value *Offset =
      llvm::ConstantInt::get(CGF.Int32Ty, (SizeInBytes + 7)  & ~7);
  overflow_arg_area = CGF.Builder.CreateGEP(overflow_arg_area, Offset,
                                            "overflow_arg_area.next");
  CGF.Builder.CreateStore(overflow_arg_area, overflow_arg_area_p);

  // AMD64-ABI 3.5.7p5: Step 11. Return the fetched type.
  return Res;
}

llvm::Value *X86_64ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  // Assume that va_list type is correct; should be pointer to LLVM type:
  // struct {
  //   i32 gp_offset;
  //   i32 fp_offset;
  //   i8* overflow_arg_area;
  //   i8* reg_save_area;
  // };
  unsigned neededInt, neededSSE;

  Ty = CGF.getContext().getCanonicalType(Ty);
  ABIArgInfo AI = classifyArgumentType(Ty, 0, neededInt, neededSSE);

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
    InRegs = llvm::ConstantInt::get(CGF.Int32Ty, 48 - neededInt * 8);
    InRegs = CGF.Builder.CreateICmpULE(gp_offset, InRegs, "fits_in_gp");
  }

  if (neededSSE) {
    fp_offset_p = CGF.Builder.CreateStructGEP(VAListAddr, 1, "fp_offset_p");
    fp_offset = CGF.Builder.CreateLoad(fp_offset_p, "fp_offset");
    llvm::Value *FitsInFP =
      llvm::ConstantInt::get(CGF.Int32Ty, 176 - neededSSE * 16);
    FitsInFP = CGF.Builder.CreateICmpULE(fp_offset, FitsInFP, "fits_in_fp");
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
  llvm::Type *LTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *RegAddr =
    CGF.Builder.CreateLoad(CGF.Builder.CreateStructGEP(VAListAddr, 3),
                           "reg_save_area");
  if (neededInt && neededSSE) {
    // FIXME: Cleanup.
    assert(AI.isDirect() && "Unexpected ABI info for mixed regs");
    llvm::StructType *ST = cast<llvm::StructType>(AI.getCoerceToType());
    llvm::Value *Tmp = CGF.CreateTempAlloca(ST);
    assert(ST->getNumElements() == 2 && "Unexpected ABI info for mixed regs");
    llvm::Type *TyLo = ST->getElementType(0);
    llvm::Type *TyHi = ST->getElementType(1);
    assert((TyLo->isFPOrFPVectorTy() ^ TyHi->isFPOrFPVectorTy()) &&
           "Unexpected ABI info for mixed regs");
    llvm::Type *PTyLo = llvm::PointerType::getUnqual(TyLo);
    llvm::Type *PTyHi = llvm::PointerType::getUnqual(TyHi);
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
  } else if (neededSSE == 1) {
    RegAddr = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    RegAddr = CGF.Builder.CreateBitCast(RegAddr,
                                        llvm::PointerType::getUnqual(LTy));
  } else {
    assert(neededSSE == 2 && "Invalid number of needed registers!");
    // SSE registers are spaced 16 bytes apart in the register save
    // area, we need to collect the two eightbytes together.
    llvm::Value *RegAddrLo = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    llvm::Value *RegAddrHi = CGF.Builder.CreateConstGEP1_32(RegAddrLo, 16);
    llvm::Type *DoubleTy = CGF.DoubleTy;
    llvm::Type *DblPtrTy =
      llvm::PointerType::getUnqual(DoubleTy);
    llvm::StructType *ST = llvm::StructType::get(DoubleTy,
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

  // AMD64-ABI 3.5.7p5: Step 5. Set:
  // l->gp_offset = l->gp_offset + num_gp * 8
  // l->fp_offset = l->fp_offset + num_fp * 16.
  if (neededInt) {
    llvm::Value *Offset = llvm::ConstantInt::get(CGF.Int32Ty, neededInt * 8);
    CGF.Builder.CreateStore(CGF.Builder.CreateAdd(gp_offset, Offset),
                            gp_offset_p);
  }
  if (neededSSE) {
    llvm::Value *Offset = llvm::ConstantInt::get(CGF.Int32Ty, neededSSE * 16);
    CGF.Builder.CreateStore(CGF.Builder.CreateAdd(fp_offset, Offset),
                            fp_offset_p);
  }
  CGF.EmitBranch(ContBlock);

  // Emit code to load the value if it was passed in memory.

  CGF.EmitBlock(InMemBlock);
  llvm::Value *MemAddr = EmitVAArgFromMemory(VAListAddr, Ty, CGF);

  // Return the appropriate result.

  CGF.EmitBlock(ContBlock);
  llvm::PHINode *ResAddr = CGF.Builder.CreatePHI(RegAddr->getType(), 2,
                                                 "vaarg.addr");
  ResAddr->addIncoming(RegAddr, InRegBlock);
  ResAddr->addIncoming(MemAddr, InMemBlock);
  return ResAddr;
}

ABIArgInfo WinX86_64ABIInfo::classify(QualType Ty) const {

  if (Ty->isVoidType())
    return ABIArgInfo::getIgnore();

  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  uint64_t Size = getContext().getTypeSize(Ty);

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    if (hasNonTrivialDestructorOrCopyConstructor(RT) ||
        RT->getDecl()->hasFlexibleArrayMember())
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    // FIXME: mingw-w64-gcc emits 128-bit struct as i128
    if (Size == 128 &&
        getContext().getTargetInfo().getTriple().getOS()
          == llvm::Triple::MinGW32)
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                          Size));

    // MS x64 ABI requirement: "Any argument that doesn't fit in 8 bytes, or is
    // not 1, 2, 4, or 8 bytes, must be passed by reference."
    if (Size <= 64 &&
        (Size & (Size - 1)) == 0)
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                          Size));

    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
  }

  if (Ty->isPromotableIntegerType())
    return ABIArgInfo::getExtend();

  return ABIArgInfo::getDirect();
}

void WinX86_64ABIInfo::computeInfo(CGFunctionInfo &FI) const {

  QualType RetTy = FI.getReturnType();
  FI.getReturnInfo() = classify(RetTy);

  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classify(it->type);
}

llvm::Value *WinX86_64ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP,
                                                       "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, 8);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}

namespace {

class NaClX86_64ABIInfo : public ABIInfo {
 public:
  NaClX86_64ABIInfo(CodeGen::CodeGenTypes &CGT, bool HasAVX)
      : ABIInfo(CGT), PInfo(CGT), NInfo(CGT, HasAVX) {}
  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
 private:
  PNaClABIInfo PInfo;  // Used for generating calls with pnaclcall callingconv.
  X86_64ABIInfo NInfo; // Used for everything else.
};

class NaClX86_64TargetCodeGenInfo : public TargetCodeGenInfo  {
 public:
  NaClX86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, bool HasAVX)
      : TargetCodeGenInfo(new NaClX86_64ABIInfo(CGT, HasAVX)) {}
};

}

void NaClX86_64ABIInfo::computeInfo(CGFunctionInfo &FI) const {
  if (FI.getASTCallingConvention() == CC_PnaclCall)
    PInfo.computeInfo(FI);
  else
    NInfo.computeInfo(FI);
}

llvm::Value *NaClX86_64ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                          CodeGenFunction &CGF) const {
  // Always use the native convention; calling pnacl-style varargs functions
  // is unuspported.
  return NInfo.EmitVAArg(VAListAddr, Ty, CGF);
}


// PowerPC-32

namespace {
class PPC32TargetCodeGenInfo : public DefaultTargetCodeGenInfo {
public:
  PPC32TargetCodeGenInfo(CodeGenTypes &CGT) : DefaultTargetCodeGenInfo(CGT) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;
};

}

bool
PPC32TargetCodeGenInfo::initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                                llvm::Value *Address) const {
  // This is calculated from the LLVM and GCC tables and verified
  // against gcc output.  AFAIK all ABIs use the same encoding.

  CodeGen::CGBuilderTy &Builder = CGF.Builder;

  llvm::IntegerType *i8 = CGF.Int8Ty;
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);
  llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);
  llvm::Value *Sixteen8 = llvm::ConstantInt::get(i8, 16);

  // 0-31: r0-31, the 4-byte general-purpose registers
  AssignToArrayRange(Builder, Address, Four8, 0, 31);

  // 32-63: fp0-31, the 8-byte floating-point registers
  AssignToArrayRange(Builder, Address, Eight8, 32, 63);

  // 64-76 are various 4-byte special-purpose registers:
  // 64: mq
  // 65: lr
  // 66: ctr
  // 67: ap
  // 68-75 cr0-7
  // 76: xer
  AssignToArrayRange(Builder, Address, Four8, 64, 76);

  // 77-108: v0-31, the 16-byte vector registers
  AssignToArrayRange(Builder, Address, Sixteen8, 77, 108);

  // 109: vrsave
  // 110: vscr
  // 111: spe_acc
  // 112: spefscr
  // 113: sfp
  AssignToArrayRange(Builder, Address, Four8, 109, 113);

  return false;
}

// PowerPC-64

namespace {
/// PPC64_SVR4_ABIInfo - The 64-bit PowerPC ELF (SVR4) ABI information.
class PPC64_SVR4_ABIInfo : public DefaultABIInfo {

public:
  PPC64_SVR4_ABIInfo(CodeGen::CodeGenTypes &CGT) : DefaultABIInfo(CGT) {}

  bool isPromotableTypeForABI(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

  // TODO: We can add more logic to computeInfo to improve performance.
  // Example: For aggregate arguments that fit in a register, we could
  // use getDirectInReg (as is done below for structs containing a single
  // floating-point value) to avoid pushing them to memory on function
  // entry.  This would require changing the logic in PPCISelLowering
  // when lowering the parameters in the caller and args in the callee.
  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it) {
      // We rely on the default argument classification for the most part.
      // One exception:  An aggregate containing a single floating-point
      // item must be passed in a register if one is available.
      const Type *T = isSingleElementStruct(it->type, getContext());
      if (T) {
        const BuiltinType *BT = T->getAs<BuiltinType>();
        if (BT && BT->isFloatingPoint()) {
          QualType QT(T, 0);
          it->info = ABIArgInfo::getDirectInReg(CGT.ConvertType(QT));
          continue;
        }
      }
      it->info = classifyArgumentType(it->type);
    }
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, 
                                 QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class PPC64_SVR4_TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  PPC64_SVR4_TargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new PPC64_SVR4_ABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;
};

class PPC64TargetCodeGenInfo : public DefaultTargetCodeGenInfo {
public:
  PPC64TargetCodeGenInfo(CodeGenTypes &CGT) : DefaultTargetCodeGenInfo(CGT) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;
};

}

// Return true if the ABI requires Ty to be passed sign- or zero-
// extended to 64 bits.
bool
PPC64_SVR4_ABIInfo::isPromotableTypeForABI(QualType Ty) const {
  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  // Promotable integer types are required to be promoted by the ABI.
  if (Ty->isPromotableIntegerType())
    return true;

  // In addition to the usual promotable integer types, we also need to
  // extend all 32-bit types, since the ABI requires promotion to 64 bits.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return true;
    default:
      break;
    }

  return false;
}

ABIArgInfo
PPC64_SVR4_ABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty)) {
    // Records with non trivial destructors/constructors should not be passed
    // by value.
    if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    return ABIArgInfo::getIndirect(0);
  }

  return (isPromotableTypeForABI(Ty) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo
PPC64_SVR4_ABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableTypeForABI(RetTy) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

// Based on ARMABIInfo::EmitVAArg, adjusted for 64-bit machine.
llvm::Value *PPC64_SVR4_ABIInfo::EmitVAArg(llvm::Value *VAListAddr,
                                           QualType Ty,
                                           CodeGenFunction &CGF) const {
  llvm::Type *BP = CGF.Int8PtrTy;
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");

  // Update the va_list pointer.
  unsigned SizeInBytes = CGF.getContext().getTypeSize(Ty) / 8;
  unsigned Offset = llvm::RoundUpToAlignment(SizeInBytes, 8);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int64Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  // If the argument is smaller than 8 bytes, it is right-adjusted in
  // its doubleword slot.  Adjust the pointer to pick it up from the
  // correct offset.
  if (SizeInBytes < 8) {
    llvm::Value *AddrAsInt = Builder.CreatePtrToInt(Addr, CGF.Int64Ty);
    AddrAsInt = Builder.CreateAdd(AddrAsInt, Builder.getInt64(8 - SizeInBytes));
    Addr = Builder.CreateIntToPtr(AddrAsInt, BP);
  }

  llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  return Builder.CreateBitCast(Addr, PTy);
}

static bool
PPC64_initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                              llvm::Value *Address) {
  // This is calculated from the LLVM and GCC tables and verified
  // against gcc output.  AFAIK all ABIs use the same encoding.

  CodeGen::CGBuilderTy &Builder = CGF.Builder;

  llvm::IntegerType *i8 = CGF.Int8Ty;
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);
  llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);
  llvm::Value *Sixteen8 = llvm::ConstantInt::get(i8, 16);

  // 0-31: r0-31, the 8-byte general-purpose registers
  AssignToArrayRange(Builder, Address, Eight8, 0, 31);

  // 32-63: fp0-31, the 8-byte floating-point registers
  AssignToArrayRange(Builder, Address, Eight8, 32, 63);

  // 64-76 are various 4-byte special-purpose registers:
  // 64: mq
  // 65: lr
  // 66: ctr
  // 67: ap
  // 68-75 cr0-7
  // 76: xer
  AssignToArrayRange(Builder, Address, Four8, 64, 76);

  // 77-108: v0-31, the 16-byte vector registers
  AssignToArrayRange(Builder, Address, Sixteen8, 77, 108);

  // 109: vrsave
  // 110: vscr
  // 111: spe_acc
  // 112: spefscr
  // 113: sfp
  AssignToArrayRange(Builder, Address, Four8, 109, 113);

  return false;
}

bool
PPC64_SVR4_TargetCodeGenInfo::initDwarfEHRegSizeTable(
  CodeGen::CodeGenFunction &CGF,
  llvm::Value *Address) const {

  return PPC64_initDwarfEHRegSizeTable(CGF, Address);
}

bool
PPC64TargetCodeGenInfo::initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                                llvm::Value *Address) const {

  return PPC64_initDwarfEHRegSizeTable(CGF, Address);
}

//===----------------------------------------------------------------------===//
// ARM ABI Implementation
//===----------------------------------------------------------------------===//

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
  ARMABIInfo(CodeGenTypes &CGT, ABIKind _Kind) : ABIInfo(CGT), Kind(_Kind) {}

  bool isEABI() const {
    StringRef Env =
      getContext().getTargetInfo().getTriple().getEnvironmentName();
    return (Env == "gnueabi" || Env == "eabi" ||
            Env == "android" || Env == "androideabi");
  }

private:
  ABIKind getABIKind() const { return Kind; }

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, int *VFPRegs,
                                  unsigned &AllocatedVFP,
                                  bool &IsHA) const;
  bool isIllegalVectorType(QualType Ty) const;

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class ARMTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  ARMTargetCodeGenInfo(CodeGenTypes &CGT, ARMABIInfo::ABIKind K)
    :TargetCodeGenInfo(new ARMABIInfo(CGT, K)) {}

  const ARMABIInfo &getABIInfo() const {
    return static_cast<const ARMABIInfo&>(TargetCodeGenInfo::getABIInfo());
  }

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    return 13;
  }

  StringRef getARCRetainAutoreleasedReturnValueMarker() const {
    return "mov\tr7, r7\t\t@ marker for objc_retainAutoreleaseReturnValue";
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const {
    llvm::Value *Four8 = llvm::ConstantInt::get(CGF.Int8Ty, 4);

    // 0-15 are the 16 integer registers.
    AssignToArrayRange(CGF.Builder, Address, Four8, 0, 15);
    return false;
  }

  unsigned getSizeOfUnwindException() const {
    if (getABIInfo().isEABI()) return 88;
    return TargetCodeGenInfo::getSizeOfUnwindException();
  }
};

}

void ARMABIInfo::computeInfo(CGFunctionInfo &FI) const {
  // To correctly handle Homogeneous Aggregate, we need to keep track of the
  // VFP registers allocated so far.
  // C.1.vfp If the argument is a VFP CPRC and there are sufficient consecutive
  // VFP registers of the appropriate type unallocated then the argument is
  // allocated to the lowest-numbered sequence of such registers.
  // C.2.vfp If the argument is a VFP CPRC then any VFP registers that are
  // unallocated are marked as unavailable. 
  unsigned AllocatedVFP = 0;
  int VFPRegs[16] = { 0 };
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it) {
    unsigned PreAllocation = AllocatedVFP;
    bool IsHA = false;
    // 6.1.2.3 There is one VFP co-processor register class using registers
    // s0-s15 (d0-d7) for passing arguments.
    const unsigned NumVFPs = 16;
    it->info = classifyArgumentType(it->type, VFPRegs, AllocatedVFP, IsHA);
    // If we do not have enough VFP registers for the HA, any VFP registers
    // that are unallocated are marked as unavailable. To achieve this, we add
    // padding of (NumVFPs - PreAllocation) floats.
    if (IsHA && AllocatedVFP > NumVFPs && PreAllocation < NumVFPs) {
      llvm::Type *PaddingTy = llvm::ArrayType::get(
          llvm::Type::getFloatTy(getVMContext()), NumVFPs - PreAllocation);
      it->info = ABIArgInfo::getExpandWithPadding(false, PaddingTy);
    }
  }

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  // Calling convention as default by an ABI.
  llvm::CallingConv::ID DefaultCC;
  if (getContext().getTargetInfo().getTriple().getEnvironmentName()=="gnueabihf")
    DefaultCC = llvm::CallingConv::ARM_AAPCS_VFP;
  else if (isEABI())
    DefaultCC = llvm::CallingConv::ARM_AAPCS;
  else
    DefaultCC = llvm::CallingConv::ARM_APCS;

  // If user did not ask for specific calling convention explicitly (e.g. via
  // pcs attribute), set effective calling convention if it's different than ABI
  // default.
  switch (getABIKind()) {
  case APCS:
    if (DefaultCC != llvm::CallingConv::ARM_APCS)
      FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_APCS);
    break;
  case AAPCS:
    if (DefaultCC != llvm::CallingConv::ARM_AAPCS)
      FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_AAPCS);
    break;
  case AAPCS_VFP:
    if (DefaultCC != llvm::CallingConv::ARM_AAPCS_VFP)
      FI.setEffectiveCallingConvention(llvm::CallingConv::ARM_AAPCS_VFP);
    break;
  }
}

/// isHomogeneousAggregate - Return true if a type is an AAPCS-VFP homogeneous
/// aggregate.  If HAMembers is non-null, the number of base elements
/// contained in the type is returned through it; this is used for the
/// recursive calls that check aggregate component types.
static bool isHomogeneousAggregate(QualType Ty, const Type *&Base,
                                   ASTContext &Context,
                                   uint64_t *HAMembers = 0) {
  uint64_t Members = 0;
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty)) {
    if (!isHomogeneousAggregate(AT->getElementType(), Base, Context, &Members))
      return false;
    Members *= AT->getSize().getZExtValue();
  } else if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    if (RD->hasFlexibleArrayMember())
      return false;

    Members = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
      const FieldDecl *FD = *i;
      uint64_t FldMembers;
      if (!isHomogeneousAggregate(FD->getType(), Base, Context, &FldMembers))
        return false;

      Members = (RD->isUnion() ?
                 std::max(Members, FldMembers) : Members + FldMembers);
    }
  } else {
    Members = 1;
    if (const ComplexType *CT = Ty->getAs<ComplexType>()) {
      Members = 2;
      Ty = CT->getElementType();
    }

    // Homogeneous aggregates for AAPCS-VFP must have base types of float,
    // double, or 64-bit or 128-bit vectors.
    if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
      if (BT->getKind() != BuiltinType::Float && 
          BT->getKind() != BuiltinType::Double &&
          BT->getKind() != BuiltinType::LongDouble)
        return false;
    } else if (const VectorType *VT = Ty->getAs<VectorType>()) {
      unsigned VecSize = Context.getTypeSize(VT);
      if (VecSize != 64 && VecSize != 128)
        return false;
    } else {
      return false;
    }

    // The base type must be the same for all members.  Vector types of the
    // same total size are treated as being equivalent here.
    const Type *TyPtr = Ty.getTypePtr();
    if (!Base)
      Base = TyPtr;
    if (Base != TyPtr &&
        (!Base->isVectorType() || !TyPtr->isVectorType() ||
         Context.getTypeSize(Base) != Context.getTypeSize(TyPtr)))
      return false;
  }

  // Homogeneous Aggregates can have at most 4 members of the base type.
  if (HAMembers)
    *HAMembers = Members;

  return (Members > 0 && Members <= 4);
}

/// markAllocatedVFPs - update VFPRegs according to the alignment and
/// number of VFP registers (unit is S register) requested.
static void markAllocatedVFPs(int *VFPRegs, unsigned &AllocatedVFP,
                              unsigned Alignment,
                              unsigned NumRequired) {
  // Early Exit.
  if (AllocatedVFP >= 16)
    return;
  // C.1.vfp If the argument is a VFP CPRC and there are sufficient consecutive
  // VFP registers of the appropriate type unallocated then the argument is
  // allocated to the lowest-numbered sequence of such registers.
  for (unsigned I = 0; I < 16; I += Alignment) {
    bool FoundSlot = true;
    for (unsigned J = I, JEnd = I + NumRequired; J < JEnd; J++)
      if (J >= 16 || VFPRegs[J]) {
         FoundSlot = false;
         break;
      }
    if (FoundSlot) {
      for (unsigned J = I, JEnd = I + NumRequired; J < JEnd; J++)
        VFPRegs[J] = 1;
      AllocatedVFP += NumRequired;
      return;
    }
  }
  // C.2.vfp If the argument is a VFP CPRC then any VFP registers that are
  // unallocated are marked as unavailable.
  for (unsigned I = 0; I < 16; I++)
    VFPRegs[I] = 1;
  AllocatedVFP = 17; // We do not have enough VFP registers.
}

ABIArgInfo ARMABIInfo::classifyArgumentType(QualType Ty, int *VFPRegs,
                                            unsigned &AllocatedVFP,
                                            bool &IsHA) const {
  // We update number of allocated VFPs according to
  // 6.1.2.1 The following argument types are VFP CPRCs:
  //   A single-precision floating-point type (including promoted
  //   half-precision types); A double-precision floating-point type;
  //   A 64-bit or 128-bit containerized vector type; Homogeneous Aggregate
  //   with a Base Type of a single- or double-precision floating-point type,
  //   64-bit containerized vectors or 128-bit containerized vectors with one
  //   to four Elements.

  // Handle illegal vector types here.
  if (isIllegalVectorType(Ty)) {
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size <= 32) {
      llvm::Type *ResType =
          llvm::Type::getInt32Ty(getVMContext());
      return ABIArgInfo::getDirect(ResType);
    }
    if (Size == 64) {
      llvm::Type *ResType = llvm::VectorType::get(
          llvm::Type::getInt32Ty(getVMContext()), 2);
      markAllocatedVFPs(VFPRegs, AllocatedVFP, 2, 2);
      return ABIArgInfo::getDirect(ResType);
    }
    if (Size == 128) {
      llvm::Type *ResType = llvm::VectorType::get(
          llvm::Type::getInt32Ty(getVMContext()), 4);
      markAllocatedVFPs(VFPRegs, AllocatedVFP, 4, 4);
      return ABIArgInfo::getDirect(ResType);
    }
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
  }
  // Update VFPRegs for legal vector types.
  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    uint64_t Size = getContext().getTypeSize(VT);
    // Size of a legal vector should be power of 2 and above 64.
    markAllocatedVFPs(VFPRegs, AllocatedVFP, Size >= 128 ? 4 : 2, Size / 32);
  }
  // Update VFPRegs for floating point types.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    if (BT->getKind() == BuiltinType::Half ||
        BT->getKind() == BuiltinType::Float)
      markAllocatedVFPs(VFPRegs, AllocatedVFP, 1, 1);
    if (BT->getKind() == BuiltinType::Double ||
        BT->getKind() == BuiltinType::LongDouble)
      markAllocatedVFPs(VFPRegs, AllocatedVFP, 2, 2);
  }

  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Ignore empty records.
  if (isEmptyRecord(getContext(), Ty, true))
    return ABIArgInfo::getIgnore();

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  if (getABIKind() == ARMABIInfo::AAPCS_VFP) {
    // Homogeneous Aggregates need to be expanded when we can fit the aggregate
    // into VFP registers.
    const Type *Base = 0;
    uint64_t Members = 0;
    if (isHomogeneousAggregate(Ty, Base, getContext(), &Members)) {
      assert(Base && "Base class should be set for homogeneous aggregate");
      // Base can be a floating-point or a vector.
      if (Base->isVectorType()) {
        // ElementSize is in number of floats.
        unsigned ElementSize = getContext().getTypeSize(Base) == 64 ? 2 : 4;
        markAllocatedVFPs(VFPRegs, AllocatedVFP, ElementSize,
                          Members * ElementSize);
      } else if (Base->isSpecificBuiltinType(BuiltinType::Float))
        markAllocatedVFPs(VFPRegs, AllocatedVFP, 1, Members);
      else {
        assert(Base->isSpecificBuiltinType(BuiltinType::Double) ||
               Base->isSpecificBuiltinType(BuiltinType::LongDouble));
        markAllocatedVFPs(VFPRegs, AllocatedVFP, 2, Members * 2);
      }
      IsHA = true;
      return ABIArgInfo::getExpand();
    }
  }

  // Support byval for ARM.
  // The ABI alignment for APCS is 4-byte and for AAPCS at least 4-byte and at
  // most 8-byte. We realign the indirect argument if type alignment is bigger
  // than ABI alignment.
  uint64_t ABIAlign = 4;
  uint64_t TyAlign = getContext().getTypeAlign(Ty) / 8;
  if (getABIKind() == ARMABIInfo::AAPCS_VFP ||
      getABIKind() == ARMABIInfo::AAPCS)
    ABIAlign = std::min(std::max(TyAlign, (uint64_t)4), (uint64_t)8);
  if (getContext().getTypeSizeInChars(Ty) > CharUnits::fromQuantity(64)) {
    return ABIArgInfo::getIndirect(0, /*ByVal=*/true,
           /*Realign=*/TyAlign > ABIAlign);
  }

  // Otherwise, pass by coercing to a structure of the appropriate size.
  llvm::Type* ElemTy;
  unsigned SizeRegs;
  // FIXME: Try to match the types of the arguments more accurately where
  // we can.
  if (getContext().getTypeAlign(Ty) <= 32) {
    ElemTy = llvm::Type::getInt32Ty(getVMContext());
    SizeRegs = (getContext().getTypeSize(Ty) + 31) / 32;
  } else {
    ElemTy = llvm::Type::getInt64Ty(getVMContext());
    SizeRegs = (getContext().getTypeSize(Ty) + 63) / 64;
  }

  llvm::Type *STy =
    llvm::StructType::get(llvm::ArrayType::get(ElemTy, SizeRegs), NULL);
  return ABIArgInfo::getDirect(STy);
}

static bool isIntegerLikeType(QualType Ty, ASTContext &Context,
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

ABIArgInfo ARMABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // Large vector types should be returned via memory.
  if (RetTy->isVectorType() && getContext().getTypeSize(RetTy) > 128)
    return ABIArgInfo::getIndirect(0);

  if (!isAggregateTypeForABI(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (isRecordWithNonTrivialDestructorOrCopyConstructor(RetTy))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Are we following APCS?
  if (getABIKind() == APCS) {
    if (isEmptyRecord(getContext(), RetTy, false))
      return ABIArgInfo::getIgnore();

    // Complex types are all returned as packed integers.
    //
    // FIXME: Consider using 2 x vector types if the back end handles them
    // correctly.
    if (RetTy->isAnyComplexType())
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                              getContext().getTypeSize(RetTy)));

    // Integer like structures are returned in r0.
    if (isIntegerLikeType(RetTy, getContext(), getVMContext())) {
      // Return in the smallest viable integer type.
      uint64_t Size = getContext().getTypeSize(RetTy);
      if (Size <= 8)
        return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
      if (Size <= 16)
        return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
      return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
    }

    // Otherwise return in memory.
    return ABIArgInfo::getIndirect(0);
  }

  // Otherwise this is an AAPCS variant.

  if (isEmptyRecord(getContext(), RetTy, true))
    return ABIArgInfo::getIgnore();

  // Check for homogeneous aggregates with AAPCS-VFP.
  if (getABIKind() == AAPCS_VFP) {
    const Type *Base = 0;
    if (isHomogeneousAggregate(RetTy, Base, getContext())) {
      assert(Base && "Base class should be set for homogeneous aggregate");
      // Homogeneous Aggregates are returned directly.
      return ABIArgInfo::getDirect();
    }
  }

  // Aggregates <= 4 bytes are returned in r0; other aggregates
  // are returned indirectly.
  uint64_t Size = getContext().getTypeSize(RetTy);
  if (Size <= 32) {
    // Return in the smallest viable integer type.
    if (Size <= 8)
      return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
    if (Size <= 16)
      return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
    return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
  }

  return ABIArgInfo::getIndirect(0);
}

/// isIllegalVector - check whether Ty is an illegal vector type.
bool ARMABIInfo::isIllegalVectorType(QualType Ty) const {
  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // Check whether VT is legal.
    unsigned NumElements = VT->getNumElements();
    uint64_t Size = getContext().getTypeSize(VT);
    // NumElements should be power of 2.
    if ((NumElements & (NumElements - 1)) != 0)
      return true;
    // Size should be greater than 32 bits.
    return Size <= 32;
  }
  return false;
}

llvm::Value *ARMABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                   CodeGenFunction &CGF) const {
  llvm::Type *BP = CGF.Int8PtrTy;
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");

  uint64_t Size = CGF.getContext().getTypeSize(Ty) / 8;
  uint64_t TyAlign = CGF.getContext().getTypeAlign(Ty) / 8;
  bool IsIndirect = false;

  // The ABI alignment for 64-bit or 128-bit vectors is 8 for AAPCS and 4 for
  // APCS. For AAPCS, the ABI alignment is at least 4-byte and at most 8-byte.
  if (getABIKind() == ARMABIInfo::AAPCS_VFP ||
      getABIKind() == ARMABIInfo::AAPCS)
    TyAlign = std::min(std::max(TyAlign, (uint64_t)4), (uint64_t)8);
  else
    TyAlign = 4;
  // Use indirect if size of the illegal vector is bigger than 16 bytes.
  if (isIllegalVectorType(Ty) && Size > 16) {
    IsIndirect = true;
    Size = 4;
    TyAlign = 4;
  }

  // Handle address alignment for ABI alignment > 4 bytes.
  if (TyAlign > 4) {
    assert((TyAlign & (TyAlign - 1)) == 0 &&
           "Alignment is not power of 2!");
    llvm::Value *AddrAsInt = Builder.CreatePtrToInt(Addr, CGF.Int32Ty);
    AddrAsInt = Builder.CreateAdd(AddrAsInt, Builder.getInt32(TyAlign - 1));
    AddrAsInt = Builder.CreateAnd(AddrAsInt, Builder.getInt32(~(TyAlign - 1)));
    Addr = Builder.CreateIntToPtr(AddrAsInt, BP, "ap.align");
  }

  uint64_t Offset =
    llvm::RoundUpToAlignment(Size, 4);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  if (IsIndirect)
    Addr = Builder.CreateLoad(Builder.CreateBitCast(Addr, BPP));
  else if (TyAlign < CGF.getContext().getTypeAlign(Ty) / 8) {
    // We can't directly cast ap.cur to pointer to a vector type, since ap.cur
    // may not be correctly aligned for the vector type. We create an aligned
    // temporary space and copy the content over from ap.cur to the temporary
    // space. This is necessary if the natural alignment of the type is greater
    // than the ABI alignment.
    llvm::Type *I8PtrTy = Builder.getInt8PtrTy();
    CharUnits CharSize = getContext().getTypeSizeInChars(Ty);
    llvm::Value *AlignedTemp = CGF.CreateTempAlloca(CGF.ConvertType(Ty),
                                                    "var.align");
    llvm::Value *Dst = Builder.CreateBitCast(AlignedTemp, I8PtrTy);
    llvm::Value *Src = Builder.CreateBitCast(Addr, I8PtrTy);
    Builder.CreateMemCpy(Dst, Src,
        llvm::ConstantInt::get(CGF.IntPtrTy, CharSize.getQuantity()),
        TyAlign, false);
    Addr = AlignedTemp; //The content is in aligned location.
  }
  llvm::Type *PTy =
    llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  return AddrTyped;
}

namespace {

class NaClARMABIInfo : public ABIInfo {
 public:
  NaClARMABIInfo(CodeGen::CodeGenTypes &CGT, ARMABIInfo::ABIKind Kind)
      : ABIInfo(CGT), PInfo(CGT), NInfo(CGT, Kind) {}
  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
 private:
  PNaClABIInfo PInfo; // Used for generating calls with pnaclcall callingconv.
  ARMABIInfo NInfo; // Used for everything else.
};

class NaClARMTargetCodeGenInfo : public TargetCodeGenInfo  {
 public:
  NaClARMTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, ARMABIInfo::ABIKind Kind)
      : TargetCodeGenInfo(new NaClARMABIInfo(CGT, Kind)) {}
};

}

void NaClARMABIInfo::computeInfo(CGFunctionInfo &FI) const {
  if (FI.getASTCallingConvention() == CC_PnaclCall)
    PInfo.computeInfo(FI);
  else
    static_cast<const ABIInfo&>(NInfo).computeInfo(FI);
}

llvm::Value *NaClARMABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  // Always use the native convention; calling pnacl-style varargs functions
  // is unsupported.
  return static_cast<const ABIInfo&>(NInfo).EmitVAArg(VAListAddr, Ty, CGF);
}

//===----------------------------------------------------------------------===//
// NVPTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CFG) const;
};

class NVPTXTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  NVPTXTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new NVPTXABIInfo(CGT)) {}
    
  virtual void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                                   CodeGen::CodeGenModule &M) const;
};

ABIArgInfo NVPTXABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();
  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);
  return ABIArgInfo::getDirect();
}

ABIArgInfo NVPTXABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

  return ABIArgInfo::getDirect();
}

void NVPTXABIInfo::computeInfo(CGFunctionInfo &FI) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classifyArgumentType(it->type);

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  // Calling convention as default by an ABI.
  // We're still using the PTX_Kernel/PTX_Device calling conventions here,
  // but we should switch to NVVM metadata later on.
  llvm::CallingConv::ID DefaultCC;
  const LangOptions &LangOpts = getContext().getLangOpts();
  if (LangOpts.OpenCL || LangOpts.CUDA) {
    // If we are in OpenCL or CUDA mode, then default to device functions
    DefaultCC = llvm::CallingConv::PTX_Device;
  } else {
    // If we are in standard C/C++ mode, use the triple to decide on the default
    StringRef Env = 
      getContext().getTargetInfo().getTriple().getEnvironmentName();
    if (Env == "device")
      DefaultCC = llvm::CallingConv::PTX_Device;
    else
      DefaultCC = llvm::CallingConv::PTX_Kernel;
  }
  FI.setEffectiveCallingConvention(DefaultCC);
   
}

llvm::Value *NVPTXABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                     CodeGenFunction &CFG) const {
  llvm_unreachable("NVPTX does not support varargs");
}

void NVPTXTargetCodeGenInfo::
SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                    CodeGen::CodeGenModule &M) const{
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) return;

  llvm::Function *F = cast<llvm::Function>(GV);

  // Perform special handling in OpenCL mode
  if (M.getLangOpts().OpenCL) {
    // Use OpenCL function attributes to set proper calling conventions
    // By default, all functions are device functions
    if (FD->hasAttr<OpenCLKernelAttr>()) {
      // OpenCL __kernel functions get a kernel calling convention
      F->setCallingConv(llvm::CallingConv::PTX_Kernel);
      // And kernel functions are not subject to inlining
      F->addFnAttr(llvm::Attributes::NoInline);
    }
  }

  // Perform special handling in CUDA mode.
  if (M.getLangOpts().CUDA) {
    // CUDA __global__ functions get a kernel calling convention.  Since
    // __global__ functions cannot be called from the device, we do not
    // need to set the noinline attribute.
    if (FD->getAttr<CUDAGlobalAttr>())
      F->setCallingConv(llvm::CallingConv::PTX_Kernel);
  }
}

}

//===----------------------------------------------------------------------===//
// MBlaze ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class MBlazeABIInfo : public ABIInfo {
public:
  MBlazeABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

  bool isPromotableIntegerType(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
         it != ie; ++it)
      it->info = classifyArgumentType(it->type);
  }

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class MBlazeTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  MBlazeTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new MBlazeABIInfo(CGT)) {}
  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const;
};

}

bool MBlazeABIInfo::isPromotableIntegerType(QualType Ty) const {
  // MBlaze ABI requires all 8 and 16 bit quantities to be extended.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Bool:
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
      return true;
    default:
      return false;
    }
  return false;
}

llvm::Value *MBlazeABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF) const {
  // FIXME: Implement
  return 0;
}


ABIArgInfo MBlazeABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();
  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableIntegerType(RetTy) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo MBlazeABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

  return (isPromotableIntegerType(Ty) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

void MBlazeTargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                                  llvm::GlobalValue *GV,
                                                  CodeGen::CodeGenModule &M)
                                                  const {
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) return;

  llvm::CallingConv::ID CC = llvm::CallingConv::C;
  if (FD->hasAttr<MBlazeInterruptHandlerAttr>())
    CC = llvm::CallingConv::MBLAZE_INTR;
  else if (FD->hasAttr<MBlazeSaveVolatilesAttr>())
    CC = llvm::CallingConv::MBLAZE_SVOL;

  if (CC != llvm::CallingConv::C) {
      // Handle 'interrupt_handler' attribute:
      llvm::Function *F = cast<llvm::Function>(GV);

      // Step 1: Set ISR calling convention.
      F->setCallingConv(CC);

      // Step 2: Add attributes goodness.
      F->addFnAttr(llvm::Attributes::NoInline);
  }

  // Step 3: Emit _interrupt_handler alias.
  if (CC == llvm::CallingConv::MBLAZE_INTR)
    new llvm::GlobalAlias(GV->getType(), llvm::Function::ExternalLinkage,
                          "_interrupt_handler", GV, &M.getModule());
}


//===----------------------------------------------------------------------===//
// MSP430 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class MSP430TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  MSP430TargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}
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
      F->addFnAttr(llvm::Attributes::NoInline);

      // Step 3: Emit ISR vector alias.
      unsigned Num = attr->getNumber() + 0xffe0;
      new llvm::GlobalAlias(GV->getType(), llvm::Function::ExternalLinkage,
                            "vector_" + Twine::utohexstr(Num),
                            GV, &M.getModule());
    }
  }
}

//===----------------------------------------------------------------------===//
// MIPS ABI Implementation.  This works for both little-endian and
// big-endian variants.
//===----------------------------------------------------------------------===//

namespace {
class MipsABIInfo : public ABIInfo {
  bool IsO32;
  unsigned MinABIStackAlignInBytes, StackAlignInBytes;
  void CoerceToIntArgs(uint64_t TySize,
                       SmallVector<llvm::Type*, 8> &ArgList) const;
  llvm::Type* HandleAggregates(QualType Ty, uint64_t TySize) const;
  llvm::Type* returnAggregateInRegs(QualType RetTy, uint64_t Size) const;
  llvm::Type* getPaddingType(uint64_t Align, uint64_t Offset) const;
public:
  MipsABIInfo(CodeGenTypes &CGT, bool _IsO32) :
    ABIInfo(CGT), IsO32(_IsO32), MinABIStackAlignInBytes(IsO32 ? 4 : 8),
    StackAlignInBytes(IsO32 ? 8 : 16) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, uint64_t &Offset) const;
  virtual void computeInfo(CGFunctionInfo &FI) const;
  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class MIPSTargetCodeGenInfo : public TargetCodeGenInfo {
  unsigned SizeOfUnwindException;
public:
  MIPSTargetCodeGenInfo(CodeGenTypes &CGT, bool IsO32)
    : TargetCodeGenInfo(new MipsABIInfo(CGT, IsO32)),
      SizeOfUnwindException(IsO32 ? 24 : 32) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const {
    return 29;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const;

  unsigned getSizeOfUnwindException() const {
    return SizeOfUnwindException;
  }
};
}

void MipsABIInfo::CoerceToIntArgs(uint64_t TySize,
                                  SmallVector<llvm::Type*, 8> &ArgList) const {
  llvm::IntegerType *IntTy =
    llvm::IntegerType::get(getVMContext(), MinABIStackAlignInBytes * 8);

  // Add (TySize / MinABIStackAlignInBytes) args of IntTy.
  for (unsigned N = TySize / (MinABIStackAlignInBytes * 8); N; --N)
    ArgList.push_back(IntTy);

  // If necessary, add one more integer type to ArgList.
  unsigned R = TySize % (MinABIStackAlignInBytes * 8);

  if (R)
    ArgList.push_back(llvm::IntegerType::get(getVMContext(), R));
}

// In N32/64, an aligned double precision floating point field is passed in
// a register.
llvm::Type* MipsABIInfo::HandleAggregates(QualType Ty, uint64_t TySize) const {
  SmallVector<llvm::Type*, 8> ArgList, IntArgList;

  if (IsO32) {
    CoerceToIntArgs(TySize, ArgList);
    return llvm::StructType::get(getVMContext(), ArgList);
  }

  if (Ty->isComplexType())
    return CGT.ConvertType(Ty);

  const RecordType *RT = Ty->getAs<RecordType>();

  // Unions/vectors are passed in integer registers.
  if (!RT || !RT->isStructureOrClassType()) {
    CoerceToIntArgs(TySize, ArgList);
    return llvm::StructType::get(getVMContext(), ArgList);
  }

  const RecordDecl *RD = RT->getDecl();
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
  assert(!(TySize % 8) && "Size of structure must be multiple of 8.");
  
  uint64_t LastOffset = 0;
  unsigned idx = 0;
  llvm::IntegerType *I64 = llvm::IntegerType::get(getVMContext(), 64);

  // Iterate over fields in the struct/class and check if there are any aligned
  // double fields.
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
       i != e; ++i, ++idx) {
    const QualType Ty = i->getType();
    const BuiltinType *BT = Ty->getAs<BuiltinType>();

    if (!BT || BT->getKind() != BuiltinType::Double)
      continue;

    uint64_t Offset = Layout.getFieldOffset(idx);
    if (Offset % 64) // Ignore doubles that are not aligned.
      continue;

    // Add ((Offset - LastOffset) / 64) args of type i64.
    for (unsigned j = (Offset - LastOffset) / 64; j > 0; --j)
      ArgList.push_back(I64);

    // Add double type.
    ArgList.push_back(llvm::Type::getDoubleTy(getVMContext()));
    LastOffset = Offset + 64;
  }

  CoerceToIntArgs(TySize - LastOffset, IntArgList);
  ArgList.append(IntArgList.begin(), IntArgList.end());

  return llvm::StructType::get(getVMContext(), ArgList);
}

llvm::Type *MipsABIInfo::getPaddingType(uint64_t Align, uint64_t Offset) const {
  assert((Offset % MinABIStackAlignInBytes) == 0);

  if ((Align - 1) & Offset)
    return llvm::IntegerType::get(getVMContext(), MinABIStackAlignInBytes * 8);

  return 0;
}

ABIArgInfo
MipsABIInfo::classifyArgumentType(QualType Ty, uint64_t &Offset) const {
  uint64_t OrigOffset = Offset;
  uint64_t TySize = getContext().getTypeSize(Ty);
  uint64_t Align = getContext().getTypeAlign(Ty) / 8;

  Align = std::min(std::max(Align, (uint64_t)MinABIStackAlignInBytes),
                   (uint64_t)StackAlignInBytes);
  Offset = llvm::RoundUpToAlignment(Offset, Align);
  Offset += llvm::RoundUpToAlignment(TySize, Align * 8) / 8;

  if (isAggregateTypeForABI(Ty) || Ty->isVectorType()) {
    // Ignore empty aggregates.
    if (TySize == 0)
      return ABIArgInfo::getIgnore();

    // Records with non trivial destructors/constructors should not be passed
    // by value.
    if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty)) {
      Offset = OrigOffset + MinABIStackAlignInBytes;
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
    }

    // If we have reached here, aggregates are passed directly by coercing to
    // another structure type. Padding is inserted if the offset of the
    // aggregate is unaligned.
    return ABIArgInfo::getDirect(HandleAggregates(Ty, TySize), 0,
                                 getPaddingType(Align, OrigOffset));
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  if (Ty->isPromotableIntegerType())
    return ABIArgInfo::getExtend();

  return ABIArgInfo::getDirect(0, 0, getPaddingType(Align, OrigOffset));
}

llvm::Type*
MipsABIInfo::returnAggregateInRegs(QualType RetTy, uint64_t Size) const {
  const RecordType *RT = RetTy->getAs<RecordType>();
  SmallVector<llvm::Type*, 8> RTList;

  if (RT && RT->isStructureOrClassType()) {
    const RecordDecl *RD = RT->getDecl();
    const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
    unsigned FieldCnt = Layout.getFieldCount();

    // N32/64 returns struct/classes in floating point registers if the
    // following conditions are met:
    // 1. The size of the struct/class is no larger than 128-bit.
    // 2. The struct/class has one or two fields all of which are floating
    //    point types.
    // 3. The offset of the first field is zero (this follows what gcc does). 
    //
    // Any other composite results are returned in integer registers.
    //
    if (FieldCnt && (FieldCnt <= 2) && !Layout.getFieldOffset(0)) {
      RecordDecl::field_iterator b = RD->field_begin(), e = RD->field_end();
      for (; b != e; ++b) {
        const BuiltinType *BT = b->getType()->getAs<BuiltinType>();

        if (!BT || !BT->isFloatingPoint())
          break;

        RTList.push_back(CGT.ConvertType(b->getType()));
      }

      if (b == e)
        return llvm::StructType::get(getVMContext(), RTList,
                                     RD->hasAttr<PackedAttr>());

      RTList.clear();
    }
  }

  CoerceToIntArgs(Size, RTList);
  return llvm::StructType::get(getVMContext(), RTList);
}

ABIArgInfo MipsABIInfo::classifyReturnType(QualType RetTy) const {
  uint64_t Size = getContext().getTypeSize(RetTy);

  if (RetTy->isVoidType() || Size == 0)
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy) || RetTy->isVectorType()) {
    if (Size <= 128) {
      if (RetTy->isAnyComplexType())
        return ABIArgInfo::getDirect();

      // O32 returns integer vectors in registers.
      if (IsO32 && RetTy->isVectorType() && !RetTy->hasFloatingRepresentation())
        return ABIArgInfo::getDirect(returnAggregateInRegs(RetTy, Size));

      if (!IsO32 && !isRecordWithNonTrivialDestructorOrCopyConstructor(RetTy))
        return ABIArgInfo::getDirect(returnAggregateInRegs(RetTy, Size));
    }

    return ABIArgInfo::getIndirect(0);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

void MipsABIInfo::computeInfo(CGFunctionInfo &FI) const {
  ABIArgInfo &RetInfo = FI.getReturnInfo();
  RetInfo = classifyReturnType(FI.getReturnType());

  // Check if a pointer to an aggregate is passed as a hidden argument.  
  uint64_t Offset = RetInfo.isIndirect() ? MinABIStackAlignInBytes : 0;

  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classifyArgumentType(it->type, Offset);
}

llvm::Value* MipsABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                    CodeGenFunction &CGF) const {
  llvm::Type *BP = CGF.Int8PtrTy;
  llvm::Type *BPP = CGF.Int8PtrPtrTy;
 
  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  int64_t TypeAlign = getContext().getTypeAlign(Ty) / 8;
  llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped;
  unsigned PtrWidth = getContext().getTargetInfo().getPointerWidth(0);
  llvm::IntegerType *IntTy = (PtrWidth == 32) ? CGF.Int32Ty : CGF.Int64Ty;

  if (TypeAlign > MinABIStackAlignInBytes) {
    llvm::Value *AddrAsInt = CGF.Builder.CreatePtrToInt(Addr, IntTy);
    llvm::Value *Inc = llvm::ConstantInt::get(IntTy, TypeAlign - 1);
    llvm::Value *Mask = llvm::ConstantInt::get(IntTy, -TypeAlign);
    llvm::Value *Add = CGF.Builder.CreateAdd(AddrAsInt, Inc);
    llvm::Value *And = CGF.Builder.CreateAnd(Add, Mask);
    AddrTyped = CGF.Builder.CreateIntToPtr(And, PTy);
  }
  else
    AddrTyped = Builder.CreateBitCast(Addr, PTy);  

  llvm::Value *AlignedAddr = Builder.CreateBitCast(AddrTyped, BP);
  TypeAlign = std::max((unsigned)TypeAlign, MinABIStackAlignInBytes);
  uint64_t Offset =
    llvm::RoundUpToAlignment(CGF.getContext().getTypeSize(Ty) / 8, TypeAlign);
  llvm::Value *NextAddr =
    Builder.CreateGEP(AlignedAddr, llvm::ConstantInt::get(IntTy, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);
  
  return AddrTyped;
}

bool
MIPSTargetCodeGenInfo::initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                               llvm::Value *Address) const {
  // This information comes from gcc's implementation, which seems to
  // as canonical as it gets.

  // Everything on MIPS is 4 bytes.  Double-precision FP registers
  // are aliased to pairs of single-precision FP registers.
  llvm::Value *Four8 = llvm::ConstantInt::get(CGF.Int8Ty, 4);

  // 0-31 are the general purpose registers, $0 - $31.
  // 32-63 are the floating-point registers, $f0 - $f31.
  // 64 and 65 are the multiply/divide registers, $hi and $lo.
  // 66 is the (notional, I think) register for signal-handler return.
  AssignToArrayRange(CGF.Builder, Address, Four8, 0, 65);

  // 67-74 are the floating-point status registers, $fcc0 - $fcc7.
  // They are one bit wide and ignored here.

  // 80-111 are the coprocessor 0 registers, $c0r0 - $c0r31.
  // (coprocessor 1 is the FP unit)
  // 112-143 are the coprocessor 2 registers, $c2r0 - $c2r31.
  // 144-175 are the coprocessor 3 registers, $c3r0 - $c3r31.
  // 176-181 are the DSP accumulator registers.
  AssignToArrayRange(CGF.Builder, Address, Four8, 80, 181);
  return false;
}

//===----------------------------------------------------------------------===//
// TCE ABI Implementation (see http://tce.cs.tut.fi). Uses mostly the defaults.
// Currently subclassed only to implement custom OpenCL C function attribute 
// handling.
//===----------------------------------------------------------------------===//

namespace {

class TCETargetCodeGenInfo : public DefaultTargetCodeGenInfo {
public:
  TCETargetCodeGenInfo(CodeGenTypes &CGT)
    : DefaultTargetCodeGenInfo(CGT) {}

  virtual void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                                   CodeGen::CodeGenModule &M) const;
};

void TCETargetCodeGenInfo::SetTargetAttributes(const Decl *D,
                                               llvm::GlobalValue *GV,
                                               CodeGen::CodeGenModule &M) const {
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) return;

  llvm::Function *F = cast<llvm::Function>(GV);
  
  if (M.getLangOpts().OpenCL) {
    if (FD->hasAttr<OpenCLKernelAttr>()) {
      // OpenCL C Kernel functions are not subject to inlining
      F->addFnAttr(llvm::Attributes::NoInline);
          
      if (FD->hasAttr<ReqdWorkGroupSizeAttr>()) {

        // Convert the reqd_work_group_size() attributes to metadata.
        llvm::LLVMContext &Context = F->getContext();
        llvm::NamedMDNode *OpenCLMetadata = 
            M.getModule().getOrInsertNamedMetadata("opencl.kernel_wg_size_info");

        SmallVector<llvm::Value*, 5> Operands;
        Operands.push_back(F);

        Operands.push_back(llvm::Constant::getIntegerValue(M.Int32Ty, 
                             llvm::APInt(32, 
                             FD->getAttr<ReqdWorkGroupSizeAttr>()->getXDim())));
        Operands.push_back(llvm::Constant::getIntegerValue(M.Int32Ty,
                             llvm::APInt(32,
                               FD->getAttr<ReqdWorkGroupSizeAttr>()->getYDim())));
        Operands.push_back(llvm::Constant::getIntegerValue(M.Int32Ty, 
                             llvm::APInt(32, 
                               FD->getAttr<ReqdWorkGroupSizeAttr>()->getZDim())));

        // Add a boolean constant operand for "required" (true) or "hint" (false)
        // for implementing the work_group_size_hint attr later. Currently 
        // always true as the hint is not yet implemented.
        Operands.push_back(llvm::ConstantInt::getTrue(Context));
        OpenCLMetadata->addOperand(llvm::MDNode::get(Context, Operands));
      }
    }
  }
}

}

//===----------------------------------------------------------------------===//
// Hexagon ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class HexagonABIInfo : public ABIInfo {


public:
  HexagonABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

private:

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  virtual void computeInfo(CGFunctionInfo &FI) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const;
};

class HexagonTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  HexagonTargetCodeGenInfo(CodeGenTypes &CGT)
    :TargetCodeGenInfo(new HexagonABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
    return 29;
  }
};

}

void HexagonABIInfo::computeInfo(CGFunctionInfo &FI) const {
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it)
    it->info = classifyArgumentType(it->type);
}

ABIArgInfo HexagonABIInfo::classifyArgumentType(QualType Ty) const {
  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Ignore empty records.
  if (isEmptyRecord(getContext(), Ty, true))
    return ABIArgInfo::getIgnore();

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (isRecordWithNonTrivialDestructorOrCopyConstructor(Ty))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  uint64_t Size = getContext().getTypeSize(Ty);
  if (Size > 64)
    return ABIArgInfo::getIndirect(0, /*ByVal=*/true);
    // Pass in the smallest viable integer type.
  else if (Size > 32)
      return ABIArgInfo::getDirect(llvm::Type::getInt64Ty(getVMContext()));
  else if (Size > 16)
      return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
  else if (Size > 8)
      return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
  else
      return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
}

ABIArgInfo HexagonABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // Large vector types should be returned via memory.
  if (RetTy->isVectorType() && getContext().getTypeSize(RetTy) > 64)
    return ABIArgInfo::getIndirect(0);

  if (!isAggregateTypeForABI(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() ?
            ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (isRecordWithNonTrivialDestructorOrCopyConstructor(RetTy))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  if (isEmptyRecord(getContext(), RetTy, true))
    return ABIArgInfo::getIgnore();

  // Aggregates <= 8 bytes are returned in r0; other aggregates
  // are returned indirectly.
  uint64_t Size = getContext().getTypeSize(RetTy);
  if (Size <= 64) {
    // Return in the smallest viable integer type.
    if (Size <= 8)
      return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
    if (Size <= 16)
      return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
    if (Size <= 32)
      return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
    return ABIArgInfo::getDirect(llvm::Type::getInt64Ty(getVMContext()));
  }

  return ABIArgInfo::getIndirect(0, /*ByVal=*/true);
}

llvm::Value *HexagonABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  // FIXME: Need to handle alignment
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

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
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  return AddrTyped;
}


const TargetCodeGenInfo &CodeGenModule::getTargetCodeGenInfo() {
  if (TheTargetCodeGenInfo)
    return *TheTargetCodeGenInfo;

  const llvm::Triple &Triple = getContext().getTargetInfo().getTriple();
  switch (Triple.getArch()) {
  default:
    return *(TheTargetCodeGenInfo = new DefaultTargetCodeGenInfo(Types));

  case llvm::Triple::le32:
    return *(TheTargetCodeGenInfo = new PNaClTargetCodeGenInfo(Types));
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return *(TheTargetCodeGenInfo = new MIPSTargetCodeGenInfo(Types, true));

  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    return *(TheTargetCodeGenInfo = new MIPSTargetCodeGenInfo(Types, false));

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    {
      ARMABIInfo::ABIKind Kind = ARMABIInfo::AAPCS;
      if (strcmp(getContext().getTargetInfo().getABI(), "apcs-gnu") == 0)
        Kind = ARMABIInfo::APCS;
      else if (CodeGenOpts.FloatABI == "hard" ||
               (CodeGenOpts.FloatABI != "soft" && Triple.getEnvironment()==llvm::Triple::GNUEABIHF))
        Kind = ARMABIInfo::AAPCS_VFP;

      switch (Triple.getOS()) {
        case llvm::Triple::NativeClient:
          return *(TheTargetCodeGenInfo =
                   new NaClARMTargetCodeGenInfo(Types, Kind));
        default:
          return *(TheTargetCodeGenInfo =
                   new ARMTargetCodeGenInfo(Types, Kind));
      }
    }

  case llvm::Triple::ppc:
    return *(TheTargetCodeGenInfo = new PPC32TargetCodeGenInfo(Types));
  case llvm::Triple::ppc64:
    if (Triple.isOSBinFormatELF())
      return *(TheTargetCodeGenInfo = new PPC64_SVR4_TargetCodeGenInfo(Types));
    else
      return *(TheTargetCodeGenInfo = new PPC64TargetCodeGenInfo(Types));

  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return *(TheTargetCodeGenInfo = new NVPTXTargetCodeGenInfo(Types));

  case llvm::Triple::mblaze:
    return *(TheTargetCodeGenInfo = new MBlazeTargetCodeGenInfo(Types));

  case llvm::Triple::msp430:
    return *(TheTargetCodeGenInfo = new MSP430TargetCodeGenInfo(Types));

  case llvm::Triple::tce:
    return *(TheTargetCodeGenInfo = new TCETargetCodeGenInfo(Types));

  case llvm::Triple::x86: {
    bool DisableMMX = strcmp(getContext().getTargetInfo().getABI(), "no-mmx") == 0;

    if (Triple.isOSDarwin())
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, true, true, DisableMMX, false,
                                           CodeGenOpts.NumRegisterParameters));

    switch (Triple.getOS()) {
    case llvm::Triple::Cygwin:
    case llvm::Triple::MinGW32:
    case llvm::Triple::AuroraUX:
    case llvm::Triple::DragonFly:
    case llvm::Triple::FreeBSD:
    case llvm::Triple::OpenBSD:
    case llvm::Triple::Bitrig:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, false, true, DisableMMX,
                                           false,
                                           CodeGenOpts.NumRegisterParameters));

    case llvm::Triple::Win32:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, false, true, DisableMMX, true,
                                           CodeGenOpts.NumRegisterParameters));

    default:
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types, false, false, DisableMMX,
                                           false,
                                           CodeGenOpts.NumRegisterParameters));
    }
  }

  case llvm::Triple::x86_64: {
    bool HasAVX = strcmp(getContext().getTargetInfo().getABI(), "avx") == 0;

    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
    case llvm::Triple::MinGW32:
    case llvm::Triple::Cygwin:
      return *(TheTargetCodeGenInfo = new WinX86_64TargetCodeGenInfo(Types));
    case llvm::Triple::NativeClient:
      return *(TheTargetCodeGenInfo = new NaClX86_64TargetCodeGenInfo(Types, HasAVX));
    default:
      return *(TheTargetCodeGenInfo = new X86_64TargetCodeGenInfo(Types,
                                                                  HasAVX));
    }
  }
  case llvm::Triple::hexagon:
    return *(TheTargetCodeGenInfo = new HexagonTargetCodeGenInfo(Types));
  }
}
