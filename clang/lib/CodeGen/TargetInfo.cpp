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
#include "CGCXXABI.h"
#include "CGValue.h"
#include "CodeGenFunction.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>    // std::sort

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
  return !CodeGenFunction::hasScalarEvaluationKind(T) ||
         T->isMemberFunctionPointerType();
}

ABIInfo::~ABIInfo() {}

static CGCXXABI::RecordArgABI getRecordArgABI(const RecordType *RT,
                                              CGCXXABI &CXXABI) {
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return CGCXXABI::RAA_Default;
  return CXXABI.getRecordArgABI(RD);
}

static CGCXXABI::RecordArgABI getRecordArgABI(QualType T,
                                              CGCXXABI &CXXABI) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return CGCXXABI::RAA_Default;
  return getRecordArgABI(RT, CXXABI);
}

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
static QualType useFirstFieldIfTransparentUnion(QualType Ty) {
  if (const RecordType *UT = Ty->getAsUnionType()) {
    const RecordDecl *UD = UT->getDecl();
    if (UD->hasAttr<TransparentUnionAttr>()) {
      assert(!UD->field_empty() && "sema created an empty transparent union");
      return UD->field_begin()->getType();
    }
  }
  return Ty;
}

CGCXXABI &ABIInfo::getCXXABI() const {
  return CGT.getCXXABI();
}

ASTContext &ABIInfo::getContext() const {
  return CGT.getContext();
}

llvm::LLVMContext &ABIInfo::getVMContext() const {
  return CGT.getLLVMContext();
}

const llvm::DataLayout &ABIInfo::getDataLayout() const {
  return CGT.getDataLayout();
}

const TargetInfo &ABIInfo::getTarget() const {
  return CGT.getTarget();
}

bool ABIInfo::isHomogeneousAggregateBaseType(QualType Ty) const {
  return false;
}

bool ABIInfo::isHomogeneousAggregateSmallEnough(const Type *Base,
                                                uint64_t Members) const {
  return false;
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
  case InAlloca:
    OS << "InAlloca Offset=" << getInAllocaFieldIndex();
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
  //   AArch64    Linux
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

void
TargetCodeGenInfo::getDependentLibraryOption(llvm::StringRef Lib,
                                             llvm::SmallString<24> &Opt) const {
  // This assumes the user is passing a library name like "rt" instead of a
  // filename like "librt.a/so", and that they don't care whether it's static or
  // dynamic.
  Opt = "-l";
  Opt += Lib;
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
    for (const auto &I : CXXRD->bases())
      if (!isEmptyRecord(Context, I.getType(), true))
        return false;

  for (const auto *I : RD->fields())
    if (!isEmptyField(Context, I, AllowArrays))
      return false;
  return true;
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
    return nullptr;

  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return nullptr;

  const Type *Found = nullptr;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &I : CXXRD->bases()) {
      // Ignore empty records.
      if (isEmptyRecord(Context, I.getType(), true))
        continue;

      // If we already found an element then this isn't a single-element struct.
      if (Found)
        return nullptr;

      // If this is non-empty and not a single element struct, the composite
      // cannot be a single element struct.
      Found = isSingleElementStruct(I.getType(), Context);
      if (!Found)
        return nullptr;
    }
  }

  // Check for single element.
  for (const auto *FD : RD->fields()) {
    QualType FT = FD->getType();

    // Ignore empty fields.
    if (isEmptyField(Context, FD, true))
      continue;

    // If we already found an element then this isn't a single-element
    // struct.
    if (Found)
      return nullptr;

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
        return nullptr;
    }
  }

  // We don't consider a struct a single-element struct if it has
  // padding beyond the element type.
  if (Found && Context.getTypeSize(Found) != Context.getTypeSize(T))
    return nullptr;

  return Found;
}

static bool is32Or64BitBasicType(QualType Ty, ASTContext &Context) {
  // Treat complex types as the element type.
  if (const ComplexType *CTy = Ty->getAs<ComplexType>())
    Ty = CTy->getElementType();

  // Check for a type which we know has a simple scalar argument-passing
  // convention without any padding.  (We're specifically looking for 32
  // and 64-bit integer and integer-equivalents, float, and double.)
  if (!Ty->getAs<BuiltinType>() && !Ty->hasPointerRepresentation() &&
      !Ty->isEnumeralType() && !Ty->isBlockPointerType())
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

  for (const auto *FD : RD->fields()) {
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

  void computeInfo(CGFunctionInfo &FI) const override {
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &I : FI.arguments())
      I.info = classifyArgumentType(I.type);
  }

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class DefaultTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  DefaultTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}
};

llvm::Value *DefaultABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  return nullptr;
}

ABIArgInfo DefaultABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0);

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
//
// This is a simplified version of the x86_32 ABI.  Arguments and return values
// are always passed on the stack.
//===----------------------------------------------------------------------===//

class PNaClABIInfo : public ABIInfo {
 public:
  PNaClABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class PNaClTargetCodeGenInfo : public TargetCodeGenInfo {
 public:
  PNaClTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
    : TargetCodeGenInfo(new PNaClABIInfo(CGT)) {}
};

void PNaClABIInfo::computeInfo(CGFunctionInfo &FI) const {
  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  for (auto &I : FI.arguments())
    I.info = classifyArgumentType(I.type);
}

llvm::Value *PNaClABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  return nullptr;
}

/// \brief Classify argument of given type \p Ty.
ABIArgInfo PNaClABIInfo::classifyArgumentType(QualType Ty) const {
  if (isAggregateTypeForABI(Ty)) {
    if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
      return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);
    return ABIArgInfo::getIndirect(0);
  } else if (const EnumType *EnumTy = Ty->getAs<EnumType>()) {
    // Treat an enum type as its underlying type.
    Ty = EnumTy->getDecl()->getIntegerType();
  } else if (Ty->isFloatingType()) {
    // Floating-point types don't go inreg.
    return ABIArgInfo::getDirect();
  }

  return (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo PNaClABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // In the PNaCl ABI we always return records/structures on the stack.
  if (isAggregateTypeForABI(RetTy))
    return ABIArgInfo::getIndirect(0);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

/// IsX86_MMXType - Return true if this is an MMX type.
bool IsX86_MMXType(llvm::Type *IRType) {
  // Return true if the type is an MMX type <2 x i32>, <4 x i16>, or <8 x i8>.
  return IRType->isVectorTy() && IRType->getPrimitiveSizeInBits() == 64 &&
    cast<llvm::VectorType>(IRType)->getElementType()->isIntegerTy() &&
    IRType->getScalarSizeInBits() != 64;
}

static llvm::Type* X86AdjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                          StringRef Constraint,
                                          llvm::Type* Ty) {
  if ((Constraint == "y" || Constraint == "&y") && Ty->isVectorTy()) {
    if (cast<llvm::VectorType>(Ty)->getBitWidth() != 64) {
      // Invalid MMX constraint
      return nullptr;
    }

    return llvm::Type::getX86_MMXTy(CGF.getLLVMContext());
  }

  // No operation needed
  return Ty;
}

/// Returns true if this type can be passed in SSE registers with the
/// X86_VectorCall calling convention. Shared between x86_32 and x86_64.
static bool isX86VectorTypeForVectorCall(ASTContext &Context, QualType Ty) {
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    if (BT->isFloatingPoint() && BT->getKind() != BuiltinType::Half)
      return true;
  } else if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // vectorcall can pass XMM, YMM, and ZMM vectors. We don't pass SSE1 MMX
    // registers specially.
    unsigned VecSize = Context.getTypeSize(VT);
    if (VecSize == 128 || VecSize == 256 || VecSize == 512)
      return true;
  }
  return false;
}

/// Returns true if this aggregate is small enough to be passed in SSE registers
/// in the X86_VectorCall calling convention. Shared between x86_32 and x86_64.
static bool isX86VectorCallAggregateSmallEnough(uint64_t NumMembers) {
  return NumMembers <= 4;
}

//===----------------------------------------------------------------------===//
// X86-32 ABI Implementation
//===----------------------------------------------------------------------===//

/// \brief Similar to llvm::CCState, but for Clang.
struct CCState {
  CCState(unsigned CC) : CC(CC), FreeRegs(0), FreeSSERegs(0) {}

  unsigned CC;
  unsigned FreeRegs;
  unsigned FreeSSERegs;
};

/// X86_32ABIInfo - The X86-32 ABI information.
class X86_32ABIInfo : public ABIInfo {
  enum Class {
    Integer,
    Float
  };

  static const unsigned MinABIStackAlignInBytes = 4;

  bool IsDarwinVectorABI;
  bool IsSmallStructInRegABI;
  bool IsWin32StructABI;
  unsigned DefaultNumRegisterParameters;

  static bool isRegisterSize(unsigned Size) {
    return (Size == 8 || Size == 16 || Size == 32 || Size == 64);
  }

  bool isHomogeneousAggregateBaseType(QualType Ty) const override {
    // FIXME: Assumes vectorcall is in use.
    return isX86VectorTypeForVectorCall(getContext(), Ty);
  }

  bool isHomogeneousAggregateSmallEnough(const Type *Ty,
                                         uint64_t NumMembers) const override {
    // FIXME: Assumes vectorcall is in use.
    return isX86VectorCallAggregateSmallEnough(NumMembers);
  }

  bool shouldReturnTypeInRegister(QualType Ty, ASTContext &Context) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ABIArgInfo getIndirectResult(QualType Ty, bool ByVal, CCState &State) const;

  ABIArgInfo getIndirectReturnResult(CCState &State) const;

  /// \brief Return the alignment to use for the given type on the stack.
  unsigned getTypeStackAlignInBytes(QualType Ty, unsigned Align) const;

  Class classify(QualType Ty) const;
  ABIArgInfo classifyReturnType(QualType RetTy, CCState &State) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, CCState &State) const;
  bool shouldUseInReg(QualType Ty, CCState &State, bool &NeedsPadding) const;

  /// \brief Rewrite the function info so that all memory arguments use
  /// inalloca.
  void rewriteWithInAlloca(CGFunctionInfo &FI) const;

  void addFieldToArgStruct(SmallVector<llvm::Type *, 6> &FrameFields,
                           unsigned &StackOffset, ABIArgInfo &Info,
                           QualType Type) const;

public:

  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;

  X86_32ABIInfo(CodeGen::CodeGenTypes &CGT, bool d, bool p, bool w,
                unsigned r)
    : ABIInfo(CGT), IsDarwinVectorABI(d), IsSmallStructInRegABI(p),
      IsWin32StructABI(w), DefaultNumRegisterParameters(r) {}
};

class X86_32TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  X86_32TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT,
      bool d, bool p, bool w, unsigned r)
    :TargetCodeGenInfo(new X86_32ABIInfo(CGT, d, p, w, r)) {}

  static bool isStructReturnInRegABI(
      const llvm::Triple &Triple, const CodeGenOptions &Opts);

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override;

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const override {
    // Darwin uses different dwarf register numbers for EH.
    if (CGM.getTarget().getTriple().isOSDarwin()) return 5;
    return 4;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override;

  llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                  StringRef Constraint,
                                  llvm::Type* Ty) const override {
    return X86AdjustInlineAsmType(CGF, Constraint, Ty);
  }

  void addReturnRegisterOutputs(CodeGenFunction &CGF, LValue ReturnValue,
                                std::string &Constraints,
                                std::vector<llvm::Type *> &ResultRegTypes,
                                std::vector<llvm::Type *> &ResultTruncRegTypes,
                                std::vector<LValue> &ResultRegDests,
                                std::string &AsmString,
                                unsigned NumOutputs) const override;

  llvm::Constant *
  getUBSanFunctionSignature(CodeGen::CodeGenModule &CGM) const override {
    unsigned Sig = (0xeb << 0) |  // jmp rel8
                   (0x06 << 8) |  //           .+0x08
                   ('F' << 16) |
                   ('T' << 24);
    return llvm::ConstantInt::get(CGM.Int32Ty, Sig);
  }

};

}

/// Rewrite input constraint references after adding some output constraints.
/// In the case where there is one output and one input and we add one output,
/// we need to replace all operand references greater than or equal to 1:
///     mov $0, $1
///     mov eax, $1
/// The result will be:
///     mov $0, $2
///     mov eax, $2
static void rewriteInputConstraintReferences(unsigned FirstIn,
                                             unsigned NumNewOuts,
                                             std::string &AsmString) {
  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  size_t Pos = 0;
  while (Pos < AsmString.size()) {
    size_t DollarStart = AsmString.find('$', Pos);
    if (DollarStart == std::string::npos)
      DollarStart = AsmString.size();
    size_t DollarEnd = AsmString.find_first_not_of('$', DollarStart);
    if (DollarEnd == std::string::npos)
      DollarEnd = AsmString.size();
    OS << StringRef(&AsmString[Pos], DollarEnd - Pos);
    Pos = DollarEnd;
    size_t NumDollars = DollarEnd - DollarStart;
    if (NumDollars % 2 != 0 && Pos < AsmString.size()) {
      // We have an operand reference.
      size_t DigitStart = Pos;
      size_t DigitEnd = AsmString.find_first_not_of("0123456789", DigitStart);
      if (DigitEnd == std::string::npos)
        DigitEnd = AsmString.size();
      StringRef OperandStr(&AsmString[DigitStart], DigitEnd - DigitStart);
      unsigned OperandIndex;
      if (!OperandStr.getAsInteger(10, OperandIndex)) {
        if (OperandIndex >= FirstIn)
          OperandIndex += NumNewOuts;
        OS << OperandIndex;
      } else {
        OS << OperandStr;
      }
      Pos = DigitEnd;
    }
  }
  AsmString = std::move(OS.str());
}

/// Add output constraints for EAX:EDX because they are return registers.
void X86_32TargetCodeGenInfo::addReturnRegisterOutputs(
    CodeGenFunction &CGF, LValue ReturnSlot, std::string &Constraints,
    std::vector<llvm::Type *> &ResultRegTypes,
    std::vector<llvm::Type *> &ResultTruncRegTypes,
    std::vector<LValue> &ResultRegDests, std::string &AsmString,
    unsigned NumOutputs) const {
  uint64_t RetWidth = CGF.getContext().getTypeSize(ReturnSlot.getType());

  // Use the EAX constraint if the width is 32 or smaller and EAX:EDX if it is
  // larger.
  if (!Constraints.empty())
    Constraints += ',';
  if (RetWidth <= 32) {
    Constraints += "={eax}";
    ResultRegTypes.push_back(CGF.Int32Ty);
  } else {
    // Use the 'A' constraint for EAX:EDX.
    Constraints += "=A";
    ResultRegTypes.push_back(CGF.Int64Ty);
  }

  // Truncate EAX or EAX:EDX to an integer of the appropriate size.
  llvm::Type *CoerceTy = llvm::IntegerType::get(CGF.getLLVMContext(), RetWidth);
  ResultTruncRegTypes.push_back(CoerceTy);

  // Coerce the integer by bitcasting the return slot pointer.
  ReturnSlot.setAddress(CGF.Builder.CreateBitCast(ReturnSlot.getAddress(),
                                                  CoerceTy->getPointerTo()));
  ResultRegDests.push_back(ReturnSlot);

  rewriteInputConstraintReferences(NumOutputs, 1, AsmString);
}

/// shouldReturnTypeInRegister - Determine if the given type should be
/// passed in a register (for the Darwin ABI).
bool X86_32ABIInfo::shouldReturnTypeInRegister(QualType Ty,
                                               ASTContext &Context) const {
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
    return shouldReturnTypeInRegister(AT->getElementType(), Context);

  // Otherwise, it must be a record type.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (!RT) return false;

  // FIXME: Traverse bases here too.

  // Structure types are passed in register if all fields would be
  // passed in a register.
  for (const auto *FD : RT->getDecl()->fields()) {
    // Empty fields are ignored.
    if (isEmptyField(Context, FD, true))
      continue;

    // Check fields recursively.
    if (!shouldReturnTypeInRegister(FD->getType(), Context))
      return false;
  }
  return true;
}

ABIArgInfo X86_32ABIInfo::getIndirectReturnResult(CCState &State) const {
  // If the return value is indirect, then the hidden argument is consuming one
  // integer register.
  if (State.FreeRegs) {
    --State.FreeRegs;
    return ABIArgInfo::getIndirectInReg(/*Align=*/0, /*ByVal=*/false);
  }
  return ABIArgInfo::getIndirect(/*Align=*/0, /*ByVal=*/false);
}

ABIArgInfo X86_32ABIInfo::classifyReturnType(QualType RetTy, CCState &State) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  const Type *Base = nullptr;
  uint64_t NumElts = 0;
  if (State.CC == llvm::CallingConv::X86_VectorCall &&
      isHomogeneousAggregate(RetTy, Base, NumElts)) {
    // The LLVM struct type for such an aggregate should lower properly.
    return ABIArgInfo::getDirect();
  }

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

      return getIndirectReturnResult(State);
    }

    return ABIArgInfo::getDirect();
  }

  if (isAggregateTypeForABI(RetTy)) {
    if (const RecordType *RT = RetTy->getAs<RecordType>()) {
      // Structures with flexible arrays are always indirect.
      if (RT->getDecl()->hasFlexibleArrayMember())
        return getIndirectReturnResult(State);
    }

    // If specified, structs and unions are always indirect.
    if (!IsSmallStructInRegABI && !RetTy->isAnyComplexType())
      return getIndirectReturnResult(State);

    // Small structures which are register sized are generally returned
    // in a register.
    if (shouldReturnTypeInRegister(RetTy, getContext())) {
      uint64_t Size = getContext().getTypeSize(RetTy);

      // As a special-case, if the struct is a "single-element" struct, and
      // the field is of type "float" or "double", return it in a
      // floating-point register. (MSVC does not apply this special case.)
      // We apply a similar transformation for pointer types to improve the
      // quality of the generated IR.
      if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
        if ((!IsWin32StructABI && SeltTy->isRealFloatingType())
            || SeltTy->hasPointerRepresentation())
          return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

      // FIXME: We should be able to narrow this integer in cases with dead
      // padding.
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),Size));
    }

    return getIndirectReturnResult(State);
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
    for (const auto &I : CXXRD->bases())
      if (!isRecordWithSSEVectorType(Context, I.getType()))
        return false;

  for (const auto *i : RD->fields()) {
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
                                            CCState &State) const {
  if (!ByVal) {
    if (State.FreeRegs) {
      --State.FreeRegs; // Non-byval indirects just use one pointer.
      return ABIArgInfo::getIndirectInReg(0, false);
    }
    return ABIArgInfo::getIndirect(0, false);
  }

  // Compute the byval alignment.
  unsigned TypeAlign = getContext().getTypeAlign(Ty) / 8;
  unsigned StackAlign = getTypeStackAlignInBytes(Ty, TypeAlign);
  if (StackAlign == 0)
    return ABIArgInfo::getIndirect(4, /*ByVal=*/true);

  // If the stack alignment is less than the type alignment, realign the
  // argument.
  bool Realign = TypeAlign > StackAlign;
  return ABIArgInfo::getIndirect(StackAlign, /*ByVal=*/true, Realign);
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

bool X86_32ABIInfo::shouldUseInReg(QualType Ty, CCState &State,
                                   bool &NeedsPadding) const {
  NeedsPadding = false;
  Class C = classify(Ty);
  if (C == Float)
    return false;

  unsigned Size = getContext().getTypeSize(Ty);
  unsigned SizeInRegs = (Size + 31) / 32;

  if (SizeInRegs == 0)
    return false;

  if (SizeInRegs > State.FreeRegs) {
    State.FreeRegs = 0;
    return false;
  }

  State.FreeRegs -= SizeInRegs;

  if (State.CC == llvm::CallingConv::X86_FastCall ||
      State.CC == llvm::CallingConv::X86_VectorCall) {
    if (Size > 32)
      return false;

    if (Ty->isIntegralOrEnumerationType())
      return true;

    if (Ty->isPointerType())
      return true;

    if (Ty->isReferenceType())
      return true;

    if (State.FreeRegs)
      NeedsPadding = true;

    return false;
  }

  return true;
}

ABIArgInfo X86_32ABIInfo::classifyArgumentType(QualType Ty,
                                               CCState &State) const {
  // FIXME: Set alignment on indirect arguments.

  Ty = useFirstFieldIfTransparentUnion(Ty);

  // Check with the C++ ABI first.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (RT) {
    CGCXXABI::RecordArgABI RAA = getRecordArgABI(RT, getCXXABI());
    if (RAA == CGCXXABI::RAA_Indirect) {
      return getIndirectResult(Ty, false, State);
    } else if (RAA == CGCXXABI::RAA_DirectInMemory) {
      // The field index doesn't matter, we'll fix it up later.
      return ABIArgInfo::getInAlloca(/*FieldIndex=*/0);
    }
  }

  // vectorcall adds the concept of a homogenous vector aggregate, similar
  // to other targets.
  const Type *Base = nullptr;
  uint64_t NumElts = 0;
  if (State.CC == llvm::CallingConv::X86_VectorCall &&
      isHomogeneousAggregate(Ty, Base, NumElts)) {
    if (State.FreeSSERegs >= NumElts) {
      State.FreeSSERegs -= NumElts;
      if (Ty->isBuiltinType() || Ty->isVectorType())
        return ABIArgInfo::getDirect();
      return ABIArgInfo::getExpand();
    }
    return getIndirectResult(Ty, /*ByVal=*/false, State);
  }

  if (isAggregateTypeForABI(Ty)) {
    if (RT) {
      // Structs are always byval on win32, regardless of what they contain.
      if (IsWin32StructABI)
        return getIndirectResult(Ty, true, State);

      // Structures with flexible arrays are always indirect.
      if (RT->getDecl()->hasFlexibleArrayMember())
        return getIndirectResult(Ty, true, State);
    }

    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();

    llvm::LLVMContext &LLVMContext = getVMContext();
    llvm::IntegerType *Int32 = llvm::Type::getInt32Ty(LLVMContext);
    bool NeedsPadding;
    if (shouldUseInReg(Ty, State, NeedsPadding)) {
      unsigned SizeInRegs = (getContext().getTypeSize(Ty) + 31) / 32;
      SmallVector<llvm::Type*, 3> Elements(SizeInRegs, Int32);
      llvm::Type *Result = llvm::StructType::get(LLVMContext, Elements);
      return ABIArgInfo::getDirectInReg(Result);
    }
    llvm::IntegerType *PaddingType = NeedsPadding ? Int32 : nullptr;

    // Expand small (<= 128-bit) record types when we know that the stack layout
    // of those arguments will match the struct. This is important because the
    // LLVM backend isn't smart enough to remove byval, which inhibits many
    // optimizations.
    if (getContext().getTypeSize(Ty) <= 4*32 &&
        canExpandIndirectArgument(Ty, getContext()))
      return ABIArgInfo::getExpandWithPadding(
          State.CC == llvm::CallingConv::X86_FastCall ||
              State.CC == llvm::CallingConv::X86_VectorCall,
          PaddingType);

    return getIndirectResult(Ty, true, State);
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

    if (IsX86_MMXType(CGT.ConvertType(Ty)))
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), 64));

    return ABIArgInfo::getDirect();
  }


  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  bool NeedsPadding;
  bool InReg = shouldUseInReg(Ty, State, NeedsPadding);

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
  CCState State(FI.getCallingConvention());
  if (State.CC == llvm::CallingConv::X86_FastCall)
    State.FreeRegs = 2;
  else if (State.CC == llvm::CallingConv::X86_VectorCall) {
    State.FreeRegs = 2;
    State.FreeSSERegs = 6;
  } else if (FI.getHasRegParm())
    State.FreeRegs = FI.getRegParm();
  else
    State.FreeRegs = DefaultNumRegisterParameters;

  if (!getCXXABI().classifyReturnType(FI)) {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), State);
  } else if (FI.getReturnInfo().isIndirect()) {
    // The C++ ABI is not aware of register usage, so we have to check if the
    // return value was sret and put it in a register ourselves if appropriate.
    if (State.FreeRegs) {
      --State.FreeRegs;  // The sret parameter consumes a register.
      FI.getReturnInfo().setInReg(true);
    }
  }

  // The chain argument effectively gives us another free register.
  if (FI.isChainCall())
    ++State.FreeRegs;

  bool UsedInAlloca = false;
  for (auto &I : FI.arguments()) {
    I.info = classifyArgumentType(I.type, State);
    UsedInAlloca |= (I.info.getKind() == ABIArgInfo::InAlloca);
  }

  // If we needed to use inalloca for any argument, do a second pass and rewrite
  // all the memory arguments to use inalloca.
  if (UsedInAlloca)
    rewriteWithInAlloca(FI);
}

void
X86_32ABIInfo::addFieldToArgStruct(SmallVector<llvm::Type *, 6> &FrameFields,
                                   unsigned &StackOffset,
                                   ABIArgInfo &Info, QualType Type) const {
  assert(StackOffset % 4U == 0 && "unaligned inalloca struct");
  Info = ABIArgInfo::getInAlloca(FrameFields.size());
  FrameFields.push_back(CGT.ConvertTypeForMem(Type));
  StackOffset += getContext().getTypeSizeInChars(Type).getQuantity();

  // Insert padding bytes to respect alignment.  For x86_32, each argument is 4
  // byte aligned.
  if (StackOffset % 4U) {
    unsigned OldOffset = StackOffset;
    StackOffset = llvm::RoundUpToAlignment(StackOffset, 4U);
    unsigned NumBytes = StackOffset - OldOffset;
    assert(NumBytes);
    llvm::Type *Ty = llvm::Type::getInt8Ty(getVMContext());
    Ty = llvm::ArrayType::get(Ty, NumBytes);
    FrameFields.push_back(Ty);
  }
}

static bool isArgInAlloca(const ABIArgInfo &Info) {
  // Leave ignored and inreg arguments alone.
  switch (Info.getKind()) {
  case ABIArgInfo::InAlloca:
    return true;
  case ABIArgInfo::Indirect:
    assert(Info.getIndirectByVal());
    return true;
  case ABIArgInfo::Ignore:
    return false;
  case ABIArgInfo::Direct:
  case ABIArgInfo::Extend:
  case ABIArgInfo::Expand:
    if (Info.getInReg())
      return false;
    return true;
  }
  llvm_unreachable("invalid enum");
}

void X86_32ABIInfo::rewriteWithInAlloca(CGFunctionInfo &FI) const {
  assert(IsWin32StructABI && "inalloca only supported on win32");

  // Build a packed struct type for all of the arguments in memory.
  SmallVector<llvm::Type *, 6> FrameFields;

  unsigned StackOffset = 0;
  CGFunctionInfo::arg_iterator I = FI.arg_begin(), E = FI.arg_end();

  // Put 'this' into the struct before 'sret', if necessary.
  bool IsThisCall =
      FI.getCallingConvention() == llvm::CallingConv::X86_ThisCall;
  ABIArgInfo &Ret = FI.getReturnInfo();
  if (Ret.isIndirect() && Ret.isSRetAfterThis() && !IsThisCall &&
      isArgInAlloca(I->info)) {
    addFieldToArgStruct(FrameFields, StackOffset, I->info, I->type);
    ++I;
  }

  // Put the sret parameter into the inalloca struct if it's in memory.
  if (Ret.isIndirect() && !Ret.getInReg()) {
    CanQualType PtrTy = getContext().getPointerType(FI.getReturnType());
    addFieldToArgStruct(FrameFields, StackOffset, Ret, PtrTy);
    // On Windows, the hidden sret parameter is always returned in eax.
    Ret.setInAllocaSRet(IsWin32StructABI);
  }

  // Skip the 'this' parameter in ecx.
  if (IsThisCall)
    ++I;

  // Put arguments passed in memory into the struct.
  for (; I != E; ++I) {
    if (isArgInAlloca(I->info))
      addFieldToArgStruct(FrameFields, StackOffset, I->info, I->type);
  }

  FI.setArgStruct(llvm::StructType::get(getVMContext(), FrameFields,
                                        /*isPacked=*/true));
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

bool X86_32TargetCodeGenInfo::isStructReturnInRegABI(
    const llvm::Triple &Triple, const CodeGenOptions &Opts) {
  assert(Triple.getArch() == llvm::Triple::x86);

  switch (Opts.getStructReturnConvention()) {
  case CodeGenOptions::SRCK_Default:
    break;
  case CodeGenOptions::SRCK_OnStack:  // -fpcc-struct-return
    return false;
  case CodeGenOptions::SRCK_InRegs:  // -freg-struct-return
    return true;
  }

  if (Triple.isOSDarwin())
    return true;

  switch (Triple.getOS()) {
  case llvm::Triple::DragonFly:
  case llvm::Triple::FreeBSD:
  case llvm::Triple::OpenBSD:
  case llvm::Triple::Bitrig:
  case llvm::Triple::Win32:
    return true;
  default:
    return false;
  }
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
      Fn->addAttributes(llvm::AttributeSet::FunctionIndex,
                      llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                              B));
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

  if (CGF.CGM.getTarget().getTriple().isOSDarwin()) {
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
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// If a word is unused its result will be NoClass; if a type should
  /// be passed in Memory then at least the classification of \arg Lo
  /// will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will
  /// also be ComplexX87.
  void classify(QualType T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg) const;

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
                                  unsigned &neededSSE,
                                  bool isNamedArg) const;

  bool IsIllegalVectorType(QualType Ty) const;

  /// The 0.98 ABI revision clarified a lot of ambiguities,
  /// unfortunately in ways that were not always consistent with
  /// certain previous compilers.  In particular, platforms which
  /// required strict binary compatibility with older versions of GCC
  /// may need to exempt themselves.
  bool honorsRevision0_98() const {
    return !getTarget().getTriple().isOSDarwin();
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
    ABIArgInfo info = classifyArgumentType(type, 0, neededInt, neededSSE,
                                           /*isNamedArg*/true);
    if (info.isDirect()) {
      llvm::Type *ty = info.getCoerceToType();
      if (llvm::VectorType *vectorTy = dyn_cast_or_null<llvm::VectorType>(ty))
        return (vectorTy->getBitWidth() > 128);
    }
    return false;
  }

  void computeInfo(CGFunctionInfo &FI) const override;

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

/// WinX86_64ABIInfo - The Windows X86_64 ABI information.
class WinX86_64ABIInfo : public ABIInfo {

  ABIArgInfo classify(QualType Ty, unsigned &FreeSSERegs,
                      bool IsReturnType) const;

public:
  WinX86_64ABIInfo(CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  void computeInfo(CGFunctionInfo &FI) const override;

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;

  bool isHomogeneousAggregateBaseType(QualType Ty) const override {
    // FIXME: Assumes vectorcall is in use.
    return isX86VectorTypeForVectorCall(getContext(), Ty);
  }

  bool isHomogeneousAggregateSmallEnough(const Type *Ty,
                                         uint64_t NumMembers) const override {
    // FIXME: Assumes vectorcall is in use.
    return isX86VectorCallAggregateSmallEnough(NumMembers);
  }
};

class X86_64TargetCodeGenInfo : public TargetCodeGenInfo {
  bool HasAVX;
public:
  X86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, bool HasAVX)
      : TargetCodeGenInfo(new X86_64ABIInfo(CGT, HasAVX)), HasAVX(HasAVX) {}

  const X86_64ABIInfo &getABIInfo() const {
    return static_cast<const X86_64ABIInfo&>(TargetCodeGenInfo::getABIInfo());
  }

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const override {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override {
    llvm::Value *Eight8 = llvm::ConstantInt::get(CGF.Int8Ty, 8);

    // 0-15 are the 16 integer registers.
    // 16 is %rip.
    AssignToArrayRange(CGF.Builder, Address, Eight8, 0, 16);
    return false;
  }

  llvm::Type* adjustInlineAsmType(CodeGen::CodeGenFunction &CGF,
                                  StringRef Constraint,
                                  llvm::Type* Ty) const override {
    return X86AdjustInlineAsmType(CGF, Constraint, Ty);
  }

  bool isNoProtoCallVariadic(const CallArgList &args,
                             const FunctionNoProtoType *fnType) const override {
    // The default CC on x86-64 sets %al to the number of SSA
    // registers used, and GCC sets this when calling an unprototyped
    // function, so we override the default behavior.  However, don't do
    // that when AVX types are involved: the ABI explicitly states it is
    // undefined, and it doesn't work in practice because of how the ABI
    // defines varargs anyway.
    if (fnType->getCallConv() == CC_C) {
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

  llvm::Constant *
  getUBSanFunctionSignature(CodeGen::CodeGenModule &CGM) const override {
    unsigned Sig = (0xeb << 0) |  // jmp rel8
                   (0x0a << 8) |  //           .+0x0c
                   ('F' << 16) |
                   ('T' << 24);
    return llvm::ConstantInt::get(CGM.Int32Ty, Sig);
  }

  unsigned getOpenMPSimdDefaultAlignment(QualType) const override {
    return HasAVX ? 32 : 16;
  }
};

static std::string qualifyWindowsLibrary(llvm::StringRef Lib) {
  // If the argument does not end in .lib, automatically add the suffix. This
  // matches the behavior of MSVC.
  std::string ArgStr = Lib;
  if (!Lib.endswith_lower(".lib"))
    ArgStr += ".lib";
  return ArgStr;
}

class WinX86_32TargetCodeGenInfo : public X86_32TargetCodeGenInfo {
public:
  WinX86_32TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT,
        bool d, bool p, bool w, unsigned RegParms)
    : X86_32TargetCodeGenInfo(CGT, d, p, w, RegParms) {}

  void getDependentLibraryOption(llvm::StringRef Lib,
                                 llvm::SmallString<24> &Opt) const override {
    Opt = "/DEFAULTLIB:";
    Opt += qualifyWindowsLibrary(Lib);
  }

  void getDetectMismatchOption(llvm::StringRef Name,
                               llvm::StringRef Value,
                               llvm::SmallString<32> &Opt) const override {
    Opt = "/FAILIFMISMATCH:\"" + Name.str() + "=" + Value.str() + "\"";
  }
};

class WinX86_64TargetCodeGenInfo : public TargetCodeGenInfo {
  bool HasAVX;
public:
  WinX86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, bool HasAVX)
    : TargetCodeGenInfo(new WinX86_64ABIInfo(CGT)), HasAVX(HasAVX) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const override {
    return 7;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override {
    llvm::Value *Eight8 = llvm::ConstantInt::get(CGF.Int8Ty, 8);

    // 0-15 are the 16 integer registers.
    // 16 is %rip.
    AssignToArrayRange(CGF.Builder, Address, Eight8, 0, 16);
    return false;
  }

  void getDependentLibraryOption(llvm::StringRef Lib,
                                 llvm::SmallString<24> &Opt) const override {
    Opt = "/DEFAULTLIB:";
    Opt += qualifyWindowsLibrary(Lib);
  }

  void getDetectMismatchOption(llvm::StringRef Name,
                               llvm::StringRef Value,
                               llvm::SmallString<32> &Opt) const override {
    Opt = "/FAILIFMISMATCH:\"" + Name.str() + "=" + Value.str() + "\"";
  }

  unsigned getOpenMPSimdDefaultAlignment(QualType) const override {
    return HasAVX ? 32 : 16;
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
                             Class &Lo, Class &Hi, bool isNamedArg) const {
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
                getTarget().getTriple().isOSNaCl())) {
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
    classify(ET->getDecl()->getIntegerType(), OffsetBase, Lo, Hi, isNamedArg);
    return;
  }

  if (Ty->hasPointerRepresentation()) {
    Current = Integer;
    return;
  }

  if (Ty->isMemberPointerType()) {
    if (Ty->isMemberFunctionPointerType()) {
      if (Has64BitPointers) {
        // If Has64BitPointers, this is an {i64, i64}, so classify both
        // Lo and Hi now.
        Lo = Hi = Integer;
      } else {
        // Otherwise, with 32-bit pointers, this is an {i32, i32}. If that
        // straddles an eightbyte boundary, Hi should be classified as well.
        uint64_t EB_FuncPtr = (OffsetBase) / 64;
        uint64_t EB_ThisAdj = (OffsetBase + 64 - 1) / 64;
        if (EB_FuncPtr != EB_ThisAdj) {
          Lo = Hi = Integer;
        } else {
          Current = Integer;
        }
      }
    } else {
      Current = Integer;
    }
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
    } else if (Size == 128 || (HasAVX && isNamedArg && Size == 256)) {
      // Arguments of 256-bits are split into four eightbyte chunks. The
      // least significant one belongs to class SSE and all the others to class
      // SSEUP. The original Lo and Hi design considers that types can't be
      // greater than 128-bits, so a 64-bit split in Hi and Lo makes sense.
      // This design isn't correct for 256-bits, but since there're no cases
      // where the upper parts would need to be inspected, avoid adding
      // complexity and just consider Hi to match the 64-256 part.
      //
      // Note that per 3.5.7 of AMD64-ABI, 256-bit args are only passed in
      // registers if they are "named", i.e. not part of the "..." of a
      // variadic function.
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
              getTarget().getTriple().isOSNaCl()))
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
      classify(AT->getElementType(), Offset, FieldLo, FieldHi, isNamedArg);
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
    if (getRecordArgABI(RT, getCXXABI()))
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
      for (const auto &I : CXXRD->bases()) {
        assert(!I.isVirtual() && !I.getType()->isDependentType() &&
               "Unexpected base class!");
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(I.getType()->getAs<RecordType>()->getDecl());

        // Classify this field.
        //
        // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate exceeds a
        // single eightbyte, each is classified separately. Each eightbyte gets
        // initialized to class NO_CLASS.
        Class FieldLo, FieldHi;
        uint64_t Offset =
          OffsetBase + getContext().toBits(Layout.getBaseClassOffset(Base));
        classify(I.getType(), Offset, FieldLo, FieldHi, isNamedArg);
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

        if (EB_Lo) {
          assert(EB_Hi == EB_Lo && "Invalid classification, type > 16 bytes.");
          FieldLo = NoClass;
          FieldHi = Integer;
        } else {
          FieldLo = Integer;
          FieldHi = EB_Hi ? Integer : NoClass;
        }
      } else
        classify(i->getType(), Offset, FieldLo, FieldHi, isNamedArg);
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

  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
    return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);

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

/// The ABI specifies that a value should be passed in a full vector XMM/YMM
/// register. Pick an LLVM IR type that will be passed as a vector register.
llvm::Type *X86_64ABIInfo::GetByteVectorType(QualType Ty) const {
  // Wrapper structs/arrays that only contain vectors are passed just like
  // vectors; strip them off if present.
  if (const Type *InnerTy = isSingleElementStruct(Ty, getContext()))
    Ty = QualType(InnerTy, 0);

  llvm::Type *IRType = CGT.ConvertType(Ty);

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
      for (const auto &I : CXXRD->bases()) {
        assert(!I.isVirtual() && !I.getType()->isDependentType() &&
               "Unexpected base class!");
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(I.getType()->getAs<RecordType>()->getDecl());

        // If the base is after the span we care about, ignore it.
        unsigned BaseOffset = Context.toBits(Layout.getBaseClassOffset(Base));
        if (BaseOffset >= EndBit) continue;

        unsigned BaseStart = BaseOffset < StartBit ? StartBit-BaseOffset :0;
        if (!BitsContainNoUserData(I.getType(), BaseStart,
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
/// SourceTy is the source-level type for the entire argument.  SourceOffset is
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
  unsigned HiStart = llvm::RoundUpToAlignment(LoSize, HiAlign);
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

  llvm::StructType *Result = llvm::StructType::get(Lo, Hi, nullptr);


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
  classify(RetTy, 0, Lo, Hi, /*isNamedArg*/ true);

  // Check some invariants.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  llvm::Type *ResType = nullptr;
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
                                    nullptr);
    break;
  }

  llvm::Type *HighPart = nullptr;
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
  QualType Ty, unsigned freeIntRegs, unsigned &neededInt, unsigned &neededSSE,
  bool isNamedArg)
  const
{
  Ty = useFirstFieldIfTransparentUnion(Ty);

  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi, isNamedArg);

  // Check some invariants.
  // FIXME: Enforce these by construction.
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  llvm::Type *ResType = nullptr;
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
    if (getRecordArgABI(Ty, getCXXABI()) == CGCXXABI::RAA_Indirect)
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

  llvm::Type *HighPart = nullptr;
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

  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  // Keep track of the number of assigned registers.
  unsigned freeIntRegs = 6, freeSSERegs = 8;

  // If the return value is indirect, then the hidden argument is consuming one
  // integer register.
  if (FI.getReturnInfo().isIndirect())
    --freeIntRegs;

  // The chain argument effectively gives us another free register.
  if (FI.isChainCall())
    ++freeIntRegs;

  unsigned NumRequiredArgs = FI.getNumRequiredArgs();
  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  unsigned ArgNo = 0;
  for (CGFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it, ++ArgNo) {
    bool IsNamedArg = ArgNo < NumRequiredArgs;

    unsigned neededInt, neededSSE;
    it->info = classifyArgumentType(it->type, freeIntRegs, neededInt,
                                    neededSSE, IsNamedArg);

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
  ABIArgInfo AI = classifyArgumentType(Ty, 0, neededInt, neededSSE, 
                                       /*isNamedArg*/false);

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

  llvm::Value *InRegs = nullptr;
  llvm::Value *gp_offset_p = nullptr, *gp_offset = nullptr;
  llvm::Value *fp_offset_p = nullptr, *fp_offset = nullptr;
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
    llvm::Value *Tmp = CGF.CreateMemTemp(Ty);
    Tmp = CGF.Builder.CreateBitCast(Tmp, ST->getPointerTo());
    assert(ST->getNumElements() == 2 && "Unexpected ABI info for mixed regs");
    llvm::Type *TyLo = ST->getElementType(0);
    llvm::Type *TyHi = ST->getElementType(1);
    assert((TyLo->isFPOrFPVectorTy() ^ TyHi->isFPOrFPVectorTy()) &&
           "Unexpected ABI info for mixed regs");
    llvm::Type *PTyLo = llvm::PointerType::getUnqual(TyLo);
    llvm::Type *PTyHi = llvm::PointerType::getUnqual(TyHi);
    llvm::Value *GPAddr = CGF.Builder.CreateGEP(RegAddr, gp_offset);
    llvm::Value *FPAddr = CGF.Builder.CreateGEP(RegAddr, fp_offset);
    llvm::Value *RegLoAddr = TyLo->isFPOrFPVectorTy() ? FPAddr : GPAddr;
    llvm::Value *RegHiAddr = TyLo->isFPOrFPVectorTy() ? GPAddr : FPAddr;
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

    // Copy to a temporary if necessary to ensure the appropriate alignment.
    std::pair<CharUnits, CharUnits> SizeAlign =
        CGF.getContext().getTypeInfoInChars(Ty);
    uint64_t TySize = SizeAlign.first.getQuantity();
    unsigned TyAlign = SizeAlign.second.getQuantity();
    if (TyAlign > 8) {
      llvm::Value *Tmp = CGF.CreateMemTemp(Ty);
      CGF.Builder.CreateMemCpy(Tmp, RegAddr, TySize, 8, false);
      RegAddr = Tmp;
    }
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
    llvm::StructType *ST = llvm::StructType::get(DoubleTy, DoubleTy, nullptr);
    llvm::Value *V, *Tmp = CGF.CreateMemTemp(Ty);
    Tmp = CGF.Builder.CreateBitCast(Tmp, ST->getPointerTo());
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

ABIArgInfo WinX86_64ABIInfo::classify(QualType Ty, unsigned &FreeSSERegs,
                                      bool IsReturnType) const {

  if (Ty->isVoidType())
    return ABIArgInfo::getIgnore();

  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  TypeInfo Info = getContext().getTypeInfo(Ty);
  uint64_t Width = Info.Width;
  unsigned Align = getContext().toCharUnitsFromBits(Info.Align).getQuantity();

  const RecordType *RT = Ty->getAs<RecordType>();
  if (RT) {
    if (!IsReturnType) {
      if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(RT, getCXXABI()))
        return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);
    }

    if (RT->getDecl()->hasFlexibleArrayMember())
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    // FIXME: mingw-w64-gcc emits 128-bit struct as i128
    if (Width == 128 && getTarget().getTriple().isWindowsGNUEnvironment())
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(),
                                                          Width));
  }

  // vectorcall adds the concept of a homogenous vector aggregate, similar to
  // other targets.
  const Type *Base = nullptr;
  uint64_t NumElts = 0;
  if (FreeSSERegs && isHomogeneousAggregate(Ty, Base, NumElts)) {
    if (FreeSSERegs >= NumElts) {
      FreeSSERegs -= NumElts;
      if (IsReturnType || Ty->isBuiltinType() || Ty->isVectorType())
        return ABIArgInfo::getDirect();
      return ABIArgInfo::getExpand();
    }
    return ABIArgInfo::getIndirect(Align, /*ByVal=*/false);
  }


  if (Ty->isMemberPointerType()) {
    // If the member pointer is represented by an LLVM int or ptr, pass it
    // directly.
    llvm::Type *LLTy = CGT.ConvertType(Ty);
    if (LLTy->isPointerTy() || LLTy->isIntegerTy())
      return ABIArgInfo::getDirect();
  }

  if (RT || Ty->isMemberPointerType()) {
    // MS x64 ABI requirement: "Any argument that doesn't fit in 8 bytes, or is
    // not 1, 2, 4, or 8 bytes, must be passed by reference."
    if (Width > 64 || !llvm::isPowerOf2_64(Width))
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    // Otherwise, coerce it to a small integer.
    return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), Width));
  }

  // Bool type is always extended to the ABI, other builtin types are not
  // extended.
  const BuiltinType *BT = Ty->getAs<BuiltinType>();
  if (BT && BT->getKind() == BuiltinType::Bool)
    return ABIArgInfo::getExtend();

  return ABIArgInfo::getDirect();
}

void WinX86_64ABIInfo::computeInfo(CGFunctionInfo &FI) const {
  bool IsVectorCall =
      FI.getCallingConvention() == llvm::CallingConv::X86_VectorCall;

  // We can use up to 4 SSE return registers with vectorcall.
  unsigned FreeSSERegs = IsVectorCall ? 4 : 0;
  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classify(FI.getReturnType(), FreeSSERegs, true);

  // We can use up to 6 SSE register parameters with vectorcall.
  FreeSSERegs = IsVectorCall ? 6 : 0;
  for (auto &I : FI.arguments())
    I.info = classify(I.type, FreeSSERegs, false);
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
  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
 private:
  PNaClABIInfo PInfo;  // Used for generating calls with pnaclcall callingconv.
  X86_64ABIInfo NInfo; // Used for everything else.
};

class NaClX86_64TargetCodeGenInfo : public TargetCodeGenInfo  {
  bool HasAVX;
 public:
   NaClX86_64TargetCodeGenInfo(CodeGen::CodeGenTypes &CGT, bool HasAVX)
       : TargetCodeGenInfo(new NaClX86_64ABIInfo(CGT, HasAVX)), HasAVX(HasAVX) {
   }
   unsigned getOpenMPSimdDefaultAlignment(QualType) const override {
     return HasAVX ? 32 : 16;
   }
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
/// PPC32_SVR4_ABIInfo - The 32-bit PowerPC ELF (SVR4) ABI information.
class PPC32_SVR4_ABIInfo : public DefaultABIInfo {
public:
  PPC32_SVR4_ABIInfo(CodeGen::CodeGenTypes &CGT) : DefaultABIInfo(CGT) {}

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class PPC32TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  PPC32TargetCodeGenInfo(CodeGenTypes &CGT) : TargetCodeGenInfo(new PPC32_SVR4_ABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override;

  unsigned getOpenMPSimdDefaultAlignment(QualType) const override {
    return 16; // Natural alignment for Altivec vectors.
  }
};

}

llvm::Value *PPC32_SVR4_ABIInfo::EmitVAArg(llvm::Value *VAListAddr,
                                           QualType Ty,
                                           CodeGenFunction &CGF) const {
  if (const ComplexType *CTy = Ty->getAs<ComplexType>()) {
    // TODO: Implement this. For now ignore.
    (void)CTy;
    return nullptr;
  }

  bool isI64 = Ty->isIntegerType() && getContext().getTypeSize(Ty) == 64;
  bool isInt = Ty->isIntegerType() || Ty->isPointerType() || Ty->isAggregateType();
  llvm::Type *CharPtr = CGF.Int8PtrTy;
  llvm::Type *CharPtrPtr = CGF.Int8PtrPtrTy;

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *GPRPtr = Builder.CreateBitCast(VAListAddr, CharPtr, "gprptr");
  llvm::Value *GPRPtrAsInt = Builder.CreatePtrToInt(GPRPtr, CGF.Int32Ty);
  llvm::Value *FPRPtrAsInt = Builder.CreateAdd(GPRPtrAsInt, Builder.getInt32(1));
  llvm::Value *FPRPtr = Builder.CreateIntToPtr(FPRPtrAsInt, CharPtr);
  llvm::Value *OverflowAreaPtrAsInt = Builder.CreateAdd(FPRPtrAsInt, Builder.getInt32(3));
  llvm::Value *OverflowAreaPtr = Builder.CreateIntToPtr(OverflowAreaPtrAsInt, CharPtrPtr);
  llvm::Value *RegsaveAreaPtrAsInt = Builder.CreateAdd(OverflowAreaPtrAsInt, Builder.getInt32(4));
  llvm::Value *RegsaveAreaPtr = Builder.CreateIntToPtr(RegsaveAreaPtrAsInt, CharPtrPtr);
  llvm::Value *GPR = Builder.CreateLoad(GPRPtr, false, "gpr");
  // Align GPR when TY is i64.
  if (isI64) {
    llvm::Value *GPRAnd = Builder.CreateAnd(GPR, Builder.getInt8(1));
    llvm::Value *CC64 = Builder.CreateICmpEQ(GPRAnd, Builder.getInt8(1));
    llvm::Value *GPRPlusOne = Builder.CreateAdd(GPR, Builder.getInt8(1));
    GPR = Builder.CreateSelect(CC64, GPRPlusOne, GPR);
  }
  llvm::Value *FPR = Builder.CreateLoad(FPRPtr, false, "fpr");
  llvm::Value *OverflowArea = Builder.CreateLoad(OverflowAreaPtr, false, "overflow_area");
  llvm::Value *OverflowAreaAsInt = Builder.CreatePtrToInt(OverflowArea, CGF.Int32Ty);
  llvm::Value *RegsaveArea = Builder.CreateLoad(RegsaveAreaPtr, false, "regsave_area");
  llvm::Value *RegsaveAreaAsInt = Builder.CreatePtrToInt(RegsaveArea, CGF.Int32Ty);

  llvm::Value *CC = Builder.CreateICmpULT(isInt ? GPR : FPR,
                                          Builder.getInt8(8), "cond");

  llvm::Value *RegConstant = Builder.CreateMul(isInt ? GPR : FPR,
                                               Builder.getInt8(isInt ? 4 : 8));

  llvm::Value *OurReg = Builder.CreateAdd(RegsaveAreaAsInt, Builder.CreateSExt(RegConstant, CGF.Int32Ty));

  if (Ty->isFloatingType())
    OurReg = Builder.CreateAdd(OurReg, Builder.getInt32(32));

  llvm::BasicBlock *UsingRegs = CGF.createBasicBlock("using_regs");
  llvm::BasicBlock *UsingOverflow = CGF.createBasicBlock("using_overflow");
  llvm::BasicBlock *Cont = CGF.createBasicBlock("cont");

  Builder.CreateCondBr(CC, UsingRegs, UsingOverflow);

  CGF.EmitBlock(UsingRegs);

  llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *Result1 = Builder.CreateIntToPtr(OurReg, PTy);
  // Increase the GPR/FPR indexes.
  if (isInt) {
    GPR = Builder.CreateAdd(GPR, Builder.getInt8(isI64 ? 2 : 1));
    Builder.CreateStore(GPR, GPRPtr);
  } else {
    FPR = Builder.CreateAdd(FPR, Builder.getInt8(1));
    Builder.CreateStore(FPR, FPRPtr);
  }
  CGF.EmitBranch(Cont);

  CGF.EmitBlock(UsingOverflow);

  // Increase the overflow area.
  llvm::Value *Result2 = Builder.CreateIntToPtr(OverflowAreaAsInt, PTy);
  OverflowAreaAsInt = Builder.CreateAdd(OverflowAreaAsInt, Builder.getInt32(isInt ? 4 : 8));
  Builder.CreateStore(Builder.CreateIntToPtr(OverflowAreaAsInt, CharPtr), OverflowAreaPtr);
  CGF.EmitBranch(Cont);

  CGF.EmitBlock(Cont);

  llvm::PHINode *Result = CGF.Builder.CreatePHI(PTy, 2, "vaarg.addr");
  Result->addIncoming(Result1, UsingRegs);
  Result->addIncoming(Result2, UsingOverflow);

  if (Ty->isAggregateType()) {
    llvm::Value *AGGPtr = Builder.CreateBitCast(Result, CharPtrPtr, "aggrptr")  ;
    return Builder.CreateLoad(AGGPtr, false, "aggr");
  }

  return Result;
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
  enum ABIKind {
    ELFv1 = 0,
    ELFv2
  };

private:
  static const unsigned GPRBits = 64;
  ABIKind Kind;

public:
  PPC64_SVR4_ABIInfo(CodeGen::CodeGenTypes &CGT, ABIKind Kind)
    : DefaultABIInfo(CGT), Kind(Kind) {}

  bool isPromotableTypeForABI(QualType Ty) const;
  bool isAlignedParamType(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

  bool isHomogeneousAggregateBaseType(QualType Ty) const override;
  bool isHomogeneousAggregateSmallEnough(const Type *Ty,
                                         uint64_t Members) const override;

  // TODO: We can add more logic to computeInfo to improve performance.
  // Example: For aggregate arguments that fit in a register, we could
  // use getDirectInReg (as is done below for structs containing a single
  // floating-point value) to avoid pushing them to memory on function
  // entry.  This would require changing the logic in PPCISelLowering
  // when lowering the parameters in the caller and args in the callee.
  void computeInfo(CGFunctionInfo &FI) const override {
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &I : FI.arguments()) {
      // We rely on the default argument classification for the most part.
      // One exception:  An aggregate containing a single floating-point
      // or vector item must be passed in a register if one is available.
      const Type *T = isSingleElementStruct(I.type, getContext());
      if (T) {
        const BuiltinType *BT = T->getAs<BuiltinType>();
        if ((T->isVectorType() && getContext().getTypeSize(T) == 128) ||
            (BT && BT->isFloatingPoint())) {
          QualType QT(T, 0);
          I.info = ABIArgInfo::getDirectInReg(CGT.ConvertType(QT));
          continue;
        }
      }
      I.info = classifyArgumentType(I.type);
    }
  }

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class PPC64_SVR4_TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  PPC64_SVR4_TargetCodeGenInfo(CodeGenTypes &CGT,
                               PPC64_SVR4_ABIInfo::ABIKind Kind)
    : TargetCodeGenInfo(new PPC64_SVR4_ABIInfo(CGT, Kind)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override;

  unsigned getOpenMPSimdDefaultAlignment(QualType) const override {
    return 16; // Natural alignment for Altivec and VSX vectors.
  }
};

class PPC64TargetCodeGenInfo : public DefaultTargetCodeGenInfo {
public:
  PPC64TargetCodeGenInfo(CodeGenTypes &CGT) : DefaultTargetCodeGenInfo(CGT) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    // This is recovered from gcc output.
    return 1; // r1 is the dedicated stack pointer
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override;

  unsigned getOpenMPSimdDefaultAlignment(QualType) const override {
    return 16; // Natural alignment for Altivec vectors.
  }
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

/// isAlignedParamType - Determine whether a type requires 16-byte
/// alignment in the parameter area.
bool
PPC64_SVR4_ABIInfo::isAlignedParamType(QualType Ty) const {
  // Complex types are passed just like their elements.
  if (const ComplexType *CTy = Ty->getAs<ComplexType>())
    Ty = CTy->getElementType();

  // Only vector types of size 16 bytes need alignment (larger types are
  // passed via reference, smaller types are not aligned).
  if (Ty->isVectorType())
    return getContext().getTypeSize(Ty) == 128;

  // For single-element float/vector structs, we consider the whole type
  // to have the same alignment requirements as its single element.
  const Type *AlignAsType = nullptr;
  const Type *EltType = isSingleElementStruct(Ty, getContext());
  if (EltType) {
    const BuiltinType *BT = EltType->getAs<BuiltinType>();
    if ((EltType->isVectorType() &&
         getContext().getTypeSize(EltType) == 128) ||
        (BT && BT->isFloatingPoint()))
      AlignAsType = EltType;
  }

  // Likewise for ELFv2 homogeneous aggregates.
  const Type *Base = nullptr;
  uint64_t Members = 0;
  if (!AlignAsType && Kind == ELFv2 &&
      isAggregateTypeForABI(Ty) && isHomogeneousAggregate(Ty, Base, Members))
    AlignAsType = Base;

  // With special case aggregates, only vector base types need alignment.
  if (AlignAsType)
    return AlignAsType->isVectorType();

  // Otherwise, we only need alignment for any aggregate type that
  // has an alignment requirement of >= 16 bytes.
  if (isAggregateTypeForABI(Ty) && getContext().getTypeAlign(Ty) >= 128)
    return true;

  return false;
}

/// isHomogeneousAggregate - Return true if a type is an ELFv2 homogeneous
/// aggregate.  Base is set to the base element type, and Members is set
/// to the number of base elements.
bool ABIInfo::isHomogeneousAggregate(QualType Ty, const Type *&Base,
                                     uint64_t &Members) const {
  if (const ConstantArrayType *AT = getContext().getAsConstantArrayType(Ty)) {
    uint64_t NElements = AT->getSize().getZExtValue();
    if (NElements == 0)
      return false;
    if (!isHomogeneousAggregate(AT->getElementType(), Base, Members))
      return false;
    Members *= NElements;
  } else if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    if (RD->hasFlexibleArrayMember())
      return false;

    Members = 0;

    // If this is a C++ record, check the bases first.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (const auto &I : CXXRD->bases()) {
        // Ignore empty records.
        if (isEmptyRecord(getContext(), I.getType(), true))
          continue;

        uint64_t FldMembers;
        if (!isHomogeneousAggregate(I.getType(), Base, FldMembers))
          return false;

        Members += FldMembers;
      }
    }

    for (const auto *FD : RD->fields()) {
      // Ignore (non-zero arrays of) empty records.
      QualType FT = FD->getType();
      while (const ConstantArrayType *AT =
             getContext().getAsConstantArrayType(FT)) {
        if (AT->getSize().getZExtValue() == 0)
          return false;
        FT = AT->getElementType();
      }
      if (isEmptyRecord(getContext(), FT, true))
        continue;

      // For compatibility with GCC, ignore empty bitfields in C++ mode.
      if (getContext().getLangOpts().CPlusPlus &&
          FD->isBitField() && FD->getBitWidthValue(getContext()) == 0)
        continue;

      uint64_t FldMembers;
      if (!isHomogeneousAggregate(FD->getType(), Base, FldMembers))
        return false;

      Members = (RD->isUnion() ?
                 std::max(Members, FldMembers) : Members + FldMembers);
    }

    if (!Base)
      return false;

    // Ensure there is no padding.
    if (getContext().getTypeSize(Base) * Members !=
        getContext().getTypeSize(Ty))
      return false;
  } else {
    Members = 1;
    if (const ComplexType *CT = Ty->getAs<ComplexType>()) {
      Members = 2;
      Ty = CT->getElementType();
    }

    // Most ABIs only support float, double, and some vector type widths.
    if (!isHomogeneousAggregateBaseType(Ty))
      return false;

    // The base type must be the same for all members.  Types that
    // agree in both total size and mode (float vs. vector) are
    // treated as being equivalent here.
    const Type *TyPtr = Ty.getTypePtr();
    if (!Base)
      Base = TyPtr;

    if (Base->isVectorType() != TyPtr->isVectorType() ||
        getContext().getTypeSize(Base) != getContext().getTypeSize(TyPtr))
      return false;
  }
  return Members > 0 && isHomogeneousAggregateSmallEnough(Base, Members);
}

bool PPC64_SVR4_ABIInfo::isHomogeneousAggregateBaseType(QualType Ty) const {
  // Homogeneous aggregates for ELFv2 must have base types of float,
  // double, long double, or 128-bit vectors.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    if (BT->getKind() == BuiltinType::Float ||
        BT->getKind() == BuiltinType::Double ||
        BT->getKind() == BuiltinType::LongDouble)
      return true;
  }
  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    if (getContext().getTypeSize(VT) == 128)
      return true;
  }
  return false;
}

bool PPC64_SVR4_ABIInfo::isHomogeneousAggregateSmallEnough(
    const Type *Base, uint64_t Members) const {
  // Vector types require one register, floating point types require one
  // or two registers depending on their size.
  uint32_t NumRegs =
      Base->isVectorType() ? 1 : (getContext().getTypeSize(Base) + 63) / 64;

  // Homogeneous Aggregates may occupy at most 8 registers.
  return Members * NumRegs <= 8;
}

ABIArgInfo
PPC64_SVR4_ABIInfo::classifyArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (Ty->isAnyComplexType())
    return ABIArgInfo::getDirect();

  // Non-Altivec vector types are passed in GPRs (smaller than 16 bytes)
  // or via reference (larger than 16 bytes).
  if (Ty->isVectorType()) {
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size > 128)
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
    else if (Size < 128) {
      llvm::Type *CoerceTy = llvm::IntegerType::get(getVMContext(), Size);
      return ABIArgInfo::getDirect(CoerceTy);
    }
  }

  if (isAggregateTypeForABI(Ty)) {
    if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
      return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);

    uint64_t ABIAlign = isAlignedParamType(Ty)? 16 : 8;
    uint64_t TyAlign = getContext().getTypeAlign(Ty) / 8;

    // ELFv2 homogeneous aggregates are passed as array types.
    const Type *Base = nullptr;
    uint64_t Members = 0;
    if (Kind == ELFv2 &&
        isHomogeneousAggregate(Ty, Base, Members)) {
      llvm::Type *BaseTy = CGT.ConvertType(QualType(Base, 0));
      llvm::Type *CoerceTy = llvm::ArrayType::get(BaseTy, Members);
      return ABIArgInfo::getDirect(CoerceTy);
    }

    // If an aggregate may end up fully in registers, we do not
    // use the ByVal method, but pass the aggregate as array.
    // This is usually beneficial since we avoid forcing the
    // back-end to store the argument to memory.
    uint64_t Bits = getContext().getTypeSize(Ty);
    if (Bits > 0 && Bits <= 8 * GPRBits) {
      llvm::Type *CoerceTy;

      // Types up to 8 bytes are passed as integer type (which will be
      // properly aligned in the argument save area doubleword).
      if (Bits <= GPRBits)
        CoerceTy = llvm::IntegerType::get(getVMContext(),
                                          llvm::RoundUpToAlignment(Bits, 8));
      // Larger types are passed as arrays, with the base type selected
      // according to the required alignment in the save area.
      else {
        uint64_t RegBits = ABIAlign * 8;
        uint64_t NumRegs = llvm::RoundUpToAlignment(Bits, RegBits) / RegBits;
        llvm::Type *RegTy = llvm::IntegerType::get(getVMContext(), RegBits);
        CoerceTy = llvm::ArrayType::get(RegTy, NumRegs);
      }

      return ABIArgInfo::getDirect(CoerceTy);
    }

    // All other aggregates are passed ByVal.
    return ABIArgInfo::getIndirect(ABIAlign, /*ByVal=*/true,
                                   /*Realign=*/TyAlign > ABIAlign);
  }

  return (isPromotableTypeForABI(Ty) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo
PPC64_SVR4_ABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (RetTy->isAnyComplexType())
    return ABIArgInfo::getDirect();

  // Non-Altivec vector types are returned in GPRs (smaller than 16 bytes)
  // or via reference (larger than 16 bytes).
  if (RetTy->isVectorType()) {
    uint64_t Size = getContext().getTypeSize(RetTy);
    if (Size > 128)
      return ABIArgInfo::getIndirect(0);
    else if (Size < 128) {
      llvm::Type *CoerceTy = llvm::IntegerType::get(getVMContext(), Size);
      return ABIArgInfo::getDirect(CoerceTy);
    }
  }

  if (isAggregateTypeForABI(RetTy)) {
    // ELFv2 homogeneous aggregates are returned as array types.
    const Type *Base = nullptr;
    uint64_t Members = 0;
    if (Kind == ELFv2 &&
        isHomogeneousAggregate(RetTy, Base, Members)) {
      llvm::Type *BaseTy = CGT.ConvertType(QualType(Base, 0));
      llvm::Type *CoerceTy = llvm::ArrayType::get(BaseTy, Members);
      return ABIArgInfo::getDirect(CoerceTy);
    }

    // ELFv2 small aggregates are returned in up to two registers.
    uint64_t Bits = getContext().getTypeSize(RetTy);
    if (Kind == ELFv2 && Bits <= 2 * GPRBits) {
      if (Bits == 0)
        return ABIArgInfo::getIgnore();

      llvm::Type *CoerceTy;
      if (Bits > GPRBits) {
        CoerceTy = llvm::IntegerType::get(getVMContext(), GPRBits);
        CoerceTy = llvm::StructType::get(CoerceTy, CoerceTy, nullptr);
      } else
        CoerceTy = llvm::IntegerType::get(getVMContext(),
                                          llvm::RoundUpToAlignment(Bits, 8));
      return ABIArgInfo::getDirect(CoerceTy);
    }

    // All other aggregates are returned indirectly.
    return ABIArgInfo::getIndirect(0);
  }

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

  // Handle types that require 16-byte alignment in the parameter save area.
  if (isAlignedParamType(Ty)) {
    llvm::Value *AddrAsInt = Builder.CreatePtrToInt(Addr, CGF.Int64Ty);
    AddrAsInt = Builder.CreateAdd(AddrAsInt, Builder.getInt64(15));
    AddrAsInt = Builder.CreateAnd(AddrAsInt, Builder.getInt64(-16));
    Addr = Builder.CreateIntToPtr(AddrAsInt, BP, "ap.align");
  }

  // Update the va_list pointer.  The pointer should be bumped by the
  // size of the object.  We can trust getTypeSize() except for a complex
  // type whose base type is smaller than a doubleword.  For these, the
  // size of the object is 16 bytes; see below for further explanation.
  unsigned SizeInBytes = CGF.getContext().getTypeSize(Ty) / 8;
  QualType BaseTy;
  unsigned CplxBaseSize = 0;

  if (const ComplexType *CTy = Ty->getAs<ComplexType>()) {
    BaseTy = CTy->getElementType();
    CplxBaseSize = CGF.getContext().getTypeSize(BaseTy) / 8;
    if (CplxBaseSize < 8)
      SizeInBytes = 16;
  }

  unsigned Offset = llvm::RoundUpToAlignment(SizeInBytes, 8);
  llvm::Value *NextAddr =
    Builder.CreateGEP(Addr, llvm::ConstantInt::get(CGF.Int64Ty, Offset),
                      "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  // If we have a complex type and the base type is smaller than 8 bytes,
  // the ABI calls for the real and imaginary parts to be right-adjusted
  // in separate doublewords.  However, Clang expects us to produce a
  // pointer to a structure with the two parts packed tightly.  So generate
  // loads of the real and imaginary parts relative to the va_list pointer,
  // and store them to a temporary structure.
  if (CplxBaseSize && CplxBaseSize < 8) {
    llvm::Value *RealAddr = Builder.CreatePtrToInt(Addr, CGF.Int64Ty);
    llvm::Value *ImagAddr = RealAddr;
    if (CGF.CGM.getDataLayout().isBigEndian()) {
      RealAddr = Builder.CreateAdd(RealAddr, Builder.getInt64(8 - CplxBaseSize));
      ImagAddr = Builder.CreateAdd(ImagAddr, Builder.getInt64(16 - CplxBaseSize));
    } else {
      ImagAddr = Builder.CreateAdd(ImagAddr, Builder.getInt64(8));
    }
    llvm::Type *PBaseTy = llvm::PointerType::getUnqual(CGF.ConvertType(BaseTy));
    RealAddr = Builder.CreateIntToPtr(RealAddr, PBaseTy);
    ImagAddr = Builder.CreateIntToPtr(ImagAddr, PBaseTy);
    llvm::Value *Real = Builder.CreateLoad(RealAddr, false, ".vareal");
    llvm::Value *Imag = Builder.CreateLoad(ImagAddr, false, ".vaimag");
    llvm::Value *Ptr = CGF.CreateTempAlloca(CGT.ConvertTypeForMem(Ty),
                                            "vacplx");
    llvm::Value *RealPtr = Builder.CreateStructGEP(Ptr, 0, ".real");
    llvm::Value *ImagPtr = Builder.CreateStructGEP(Ptr, 1, ".imag");
    Builder.CreateStore(Real, RealPtr, false);
    Builder.CreateStore(Imag, ImagPtr, false);
    return Ptr;
  }

  // If the argument is smaller than 8 bytes, it is right-adjusted in
  // its doubleword slot.  Adjust the pointer to pick it up from the
  // correct offset.
  if (SizeInBytes < 8 && CGF.CGM.getDataLayout().isBigEndian()) {
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
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AArch64ABIInfo : public ABIInfo {
public:
  enum ABIKind {
    AAPCS = 0,
    DarwinPCS
  };

private:
  ABIKind Kind;

public:
  AArch64ABIInfo(CodeGenTypes &CGT, ABIKind Kind) : ABIInfo(CGT), Kind(Kind) {}

private:
  ABIKind getABIKind() const { return Kind; }
  bool isDarwinPCS() const { return Kind == DarwinPCS; }

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;
  bool isHomogeneousAggregateBaseType(QualType Ty) const override;
  bool isHomogeneousAggregateSmallEnough(const Type *Ty,
                                         uint64_t Members) const override;

  bool isIllegalVectorType(QualType Ty) const;

  void computeInfo(CGFunctionInfo &FI) const override {
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    for (auto &it : FI.arguments())
      it.info = classifyArgumentType(it.type);
  }

  llvm::Value *EmitDarwinVAArg(llvm::Value *VAListAddr, QualType Ty,
                               CodeGenFunction &CGF) const;

  llvm::Value *EmitAAPCSVAArg(llvm::Value *VAListAddr, QualType Ty,
                              CodeGenFunction &CGF) const;

  virtual llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                 CodeGenFunction &CGF) const override {
    return isDarwinPCS() ? EmitDarwinVAArg(VAListAddr, Ty, CGF)
                         : EmitAAPCSVAArg(VAListAddr, Ty, CGF);
  }
};

class AArch64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  AArch64TargetCodeGenInfo(CodeGenTypes &CGT, AArch64ABIInfo::ABIKind Kind)
      : TargetCodeGenInfo(new AArch64ABIInfo(CGT, Kind)) {}

  StringRef getARCRetainAutoreleasedReturnValueMarker() const {
    return "mov\tfp, fp\t\t; marker for objc_retainAutoreleaseReturnValue";
  }

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const { return 31; }

  virtual bool doesReturnSlotInterfereWithArgs() const { return false; }
};
}

ABIArgInfo AArch64ABIInfo::classifyArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  // Handle illegal vector types here.
  if (isIllegalVectorType(Ty)) {
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size <= 32) {
      llvm::Type *ResType = llvm::Type::getInt32Ty(getVMContext());
      return ABIArgInfo::getDirect(ResType);
    }
    if (Size == 64) {
      llvm::Type *ResType =
          llvm::VectorType::get(llvm::Type::getInt32Ty(getVMContext()), 2);
      return ABIArgInfo::getDirect(ResType);
    }
    if (Size == 128) {
      llvm::Type *ResType =
          llvm::VectorType::get(llvm::Type::getInt32Ty(getVMContext()), 4);
      return ABIArgInfo::getDirect(ResType);
    }
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
  }

  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    return (Ty->isPromotableIntegerType() && isDarwinPCS()
                ? ABIArgInfo::getExtend()
                : ABIArgInfo::getDirect());
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI())) {
    return ABIArgInfo::getIndirect(0, /*ByVal=*/RAA ==
                                   CGCXXABI::RAA_DirectInMemory);
  }

  // Empty records are always ignored on Darwin, but actually passed in C++ mode
  // elsewhere for GNU compatibility.
  if (isEmptyRecord(getContext(), Ty, true)) {
    if (!getContext().getLangOpts().CPlusPlus || isDarwinPCS())
      return ABIArgInfo::getIgnore();

    return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
  }

  // Homogeneous Floating-point Aggregates (HFAs) need to be expanded.
  const Type *Base = nullptr;
  uint64_t Members = 0;
  if (isHomogeneousAggregate(Ty, Base, Members)) {
    return ABIArgInfo::getDirect(
        llvm::ArrayType::get(CGT.ConvertType(QualType(Base, 0)), Members));
  }

  // Aggregates <= 16 bytes are passed directly in registers or on the stack.
  uint64_t Size = getContext().getTypeSize(Ty);
  if (Size <= 128) {
    unsigned Alignment = getContext().getTypeAlign(Ty);
    Size = 64 * ((Size + 63) / 64); // round up to multiple of 8 bytes

    // We use a pair of i64 for 16-byte aggregate with 8-byte alignment.
    // For aggregates with 16-byte alignment, we use i128.
    if (Alignment < 128 && Size == 128) {
      llvm::Type *BaseTy = llvm::Type::getInt64Ty(getVMContext());
      return ABIArgInfo::getDirect(llvm::ArrayType::get(BaseTy, Size / 64));
    }
    return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), Size));
  }

  return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
}

ABIArgInfo AArch64ABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // Large vector types should be returned via memory.
  if (RetTy->isVectorType() && getContext().getTypeSize(RetTy) > 128)
    return ABIArgInfo::getIndirect(0);

  if (!isAggregateTypeForABI(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return (RetTy->isPromotableIntegerType() && isDarwinPCS()
                ? ABIArgInfo::getExtend()
                : ABIArgInfo::getDirect());
  }

  if (isEmptyRecord(getContext(), RetTy, true))
    return ABIArgInfo::getIgnore();

  const Type *Base = nullptr;
  uint64_t Members = 0;
  if (isHomogeneousAggregate(RetTy, Base, Members))
    // Homogeneous Floating-point Aggregates (HFAs) are returned directly.
    return ABIArgInfo::getDirect();

  // Aggregates <= 16 bytes are returned directly in registers or on the stack.
  uint64_t Size = getContext().getTypeSize(RetTy);
  if (Size <= 128) {
    Size = 64 * ((Size + 63) / 64); // round up to multiple of 8 bytes
    return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), Size));
  }

  return ABIArgInfo::getIndirect(0);
}

/// isIllegalVectorType - check whether the vector type is legal for AArch64.
bool AArch64ABIInfo::isIllegalVectorType(QualType Ty) const {
  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // Check whether VT is legal.
    unsigned NumElements = VT->getNumElements();
    uint64_t Size = getContext().getTypeSize(VT);
    // NumElements should be power of 2 between 1 and 16.
    if ((NumElements & (NumElements - 1)) != 0 || NumElements > 16)
      return true;
    return Size != 64 && (Size != 128 || NumElements == 1);
  }
  return false;
}

bool AArch64ABIInfo::isHomogeneousAggregateBaseType(QualType Ty) const {
  // Homogeneous aggregates for AAPCS64 must have base types of a floating
  // point type or a short-vector type. This is the same as the 32-bit ABI,
  // but with the difference that any floating-point type is allowed,
  // including __fp16.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    if (BT->isFloatingPoint())
      return true;
  } else if (const VectorType *VT = Ty->getAs<VectorType>()) {
    unsigned VecSize = getContext().getTypeSize(VT);
    if (VecSize == 64 || VecSize == 128)
      return true;
  }
  return false;
}

bool AArch64ABIInfo::isHomogeneousAggregateSmallEnough(const Type *Base,
                                                       uint64_t Members) const {
  return Members <= 4;
}

llvm::Value *AArch64ABIInfo::EmitAAPCSVAArg(llvm::Value *VAListAddr,
                                            QualType Ty,
                                            CodeGenFunction &CGF) const {
  ABIArgInfo AI = classifyArgumentType(Ty);
  bool IsIndirect = AI.isIndirect();

  llvm::Type *BaseTy = CGF.ConvertType(Ty);
  if (IsIndirect)
    BaseTy = llvm::PointerType::getUnqual(BaseTy);
  else if (AI.getCoerceToType())
    BaseTy = AI.getCoerceToType();

  unsigned NumRegs = 1;
  if (llvm::ArrayType *ArrTy = dyn_cast<llvm::ArrayType>(BaseTy)) {
    BaseTy = ArrTy->getElementType();
    NumRegs = ArrTy->getNumElements();
  }
  bool IsFPR = BaseTy->isFloatingPointTy() || BaseTy->isVectorTy();

  // The AArch64 va_list type and handling is specified in the Procedure Call
  // Standard, section B.4:
  //
  // struct {
  //   void *__stack;
  //   void *__gr_top;
  //   void *__vr_top;
  //   int __gr_offs;
  //   int __vr_offs;
  // };

  llvm::BasicBlock *MaybeRegBlock = CGF.createBasicBlock("vaarg.maybe_reg");
  llvm::BasicBlock *InRegBlock = CGF.createBasicBlock("vaarg.in_reg");
  llvm::BasicBlock *OnStackBlock = CGF.createBasicBlock("vaarg.on_stack");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("vaarg.end");
  auto &Ctx = CGF.getContext();

  llvm::Value *reg_offs_p = nullptr, *reg_offs = nullptr;
  int reg_top_index;
  int RegSize = IsIndirect ? 8 : getContext().getTypeSize(Ty) / 8;
  if (!IsFPR) {
    // 3 is the field number of __gr_offs
    reg_offs_p = CGF.Builder.CreateStructGEP(VAListAddr, 3, "gr_offs_p");
    reg_offs = CGF.Builder.CreateLoad(reg_offs_p, "gr_offs");
    reg_top_index = 1; // field number for __gr_top
    RegSize = llvm::RoundUpToAlignment(RegSize, 8);
  } else {
    // 4 is the field number of __vr_offs.
    reg_offs_p = CGF.Builder.CreateStructGEP(VAListAddr, 4, "vr_offs_p");
    reg_offs = CGF.Builder.CreateLoad(reg_offs_p, "vr_offs");
    reg_top_index = 2; // field number for __vr_top
    RegSize = 16 * NumRegs;
  }

  //=======================================
  // Find out where argument was passed
  //=======================================

  // If reg_offs >= 0 we're already using the stack for this type of
  // argument. We don't want to keep updating reg_offs (in case it overflows,
  // though anyone passing 2GB of arguments, each at most 16 bytes, deserves
  // whatever they get).
  llvm::Value *UsingStack = nullptr;
  UsingStack = CGF.Builder.CreateICmpSGE(
      reg_offs, llvm::ConstantInt::get(CGF.Int32Ty, 0));

  CGF.Builder.CreateCondBr(UsingStack, OnStackBlock, MaybeRegBlock);

  // Otherwise, at least some kind of argument could go in these registers, the
  // question is whether this particular type is too big.
  CGF.EmitBlock(MaybeRegBlock);

  // Integer arguments may need to correct register alignment (for example a
  // "struct { __int128 a; };" gets passed in x_2N, x_{2N+1}). In this case we
  // align __gr_offs to calculate the potential address.
  if (!IsFPR && !IsIndirect && Ctx.getTypeAlign(Ty) > 64) {
    int Align = Ctx.getTypeAlign(Ty) / 8;

    reg_offs = CGF.Builder.CreateAdd(
        reg_offs, llvm::ConstantInt::get(CGF.Int32Ty, Align - 1),
        "align_regoffs");
    reg_offs = CGF.Builder.CreateAnd(
        reg_offs, llvm::ConstantInt::get(CGF.Int32Ty, -Align),
        "aligned_regoffs");
  }

  // Update the gr_offs/vr_offs pointer for next call to va_arg on this va_list.
  llvm::Value *NewOffset = nullptr;
  NewOffset = CGF.Builder.CreateAdd(
      reg_offs, llvm::ConstantInt::get(CGF.Int32Ty, RegSize), "new_reg_offs");
  CGF.Builder.CreateStore(NewOffset, reg_offs_p);

  // Now we're in a position to decide whether this argument really was in
  // registers or not.
  llvm::Value *InRegs = nullptr;
  InRegs = CGF.Builder.CreateICmpSLE(
      NewOffset, llvm::ConstantInt::get(CGF.Int32Ty, 0), "inreg");

  CGF.Builder.CreateCondBr(InRegs, InRegBlock, OnStackBlock);

  //=======================================
  // Argument was in registers
  //=======================================

  // Now we emit the code for if the argument was originally passed in
  // registers. First start the appropriate block:
  CGF.EmitBlock(InRegBlock);

  llvm::Value *reg_top_p = nullptr, *reg_top = nullptr;
  reg_top_p =
      CGF.Builder.CreateStructGEP(VAListAddr, reg_top_index, "reg_top_p");
  reg_top = CGF.Builder.CreateLoad(reg_top_p, "reg_top");
  llvm::Value *BaseAddr = CGF.Builder.CreateGEP(reg_top, reg_offs);
  llvm::Value *RegAddr = nullptr;
  llvm::Type *MemTy = llvm::PointerType::getUnqual(CGF.ConvertTypeForMem(Ty));

  if (IsIndirect) {
    // If it's been passed indirectly (actually a struct), whatever we find from
    // stored registers or on the stack will actually be a struct **.
    MemTy = llvm::PointerType::getUnqual(MemTy);
  }

  const Type *Base = nullptr;
  uint64_t NumMembers = 0;
  bool IsHFA = isHomogeneousAggregate(Ty, Base, NumMembers);
  if (IsHFA && NumMembers > 1) {
    // Homogeneous aggregates passed in registers will have their elements split
    // and stored 16-bytes apart regardless of size (they're notionally in qN,
    // qN+1, ...). We reload and store into a temporary local variable
    // contiguously.
    assert(!IsIndirect && "Homogeneous aggregates should be passed directly");
    llvm::Type *BaseTy = CGF.ConvertType(QualType(Base, 0));
    llvm::Type *HFATy = llvm::ArrayType::get(BaseTy, NumMembers);
    llvm::Value *Tmp = CGF.CreateTempAlloca(HFATy);
    int Offset = 0;

    if (CGF.CGM.getDataLayout().isBigEndian() && Ctx.getTypeSize(Base) < 128)
      Offset = 16 - Ctx.getTypeSize(Base) / 8;
    for (unsigned i = 0; i < NumMembers; ++i) {
      llvm::Value *BaseOffset =
          llvm::ConstantInt::get(CGF.Int32Ty, 16 * i + Offset);
      llvm::Value *LoadAddr = CGF.Builder.CreateGEP(BaseAddr, BaseOffset);
      LoadAddr = CGF.Builder.CreateBitCast(
          LoadAddr, llvm::PointerType::getUnqual(BaseTy));
      llvm::Value *StoreAddr = CGF.Builder.CreateStructGEP(Tmp, i);

      llvm::Value *Elem = CGF.Builder.CreateLoad(LoadAddr);
      CGF.Builder.CreateStore(Elem, StoreAddr);
    }

    RegAddr = CGF.Builder.CreateBitCast(Tmp, MemTy);
  } else {
    // Otherwise the object is contiguous in memory
    unsigned BeAlign = reg_top_index == 2 ? 16 : 8;
    if (CGF.CGM.getDataLayout().isBigEndian() &&
        (IsHFA || !isAggregateTypeForABI(Ty)) &&
        Ctx.getTypeSize(Ty) < (BeAlign * 8)) {
      int Offset = BeAlign - Ctx.getTypeSize(Ty) / 8;
      BaseAddr = CGF.Builder.CreatePtrToInt(BaseAddr, CGF.Int64Ty);

      BaseAddr = CGF.Builder.CreateAdd(
          BaseAddr, llvm::ConstantInt::get(CGF.Int64Ty, Offset), "align_be");

      BaseAddr = CGF.Builder.CreateIntToPtr(BaseAddr, CGF.Int8PtrTy);
    }

    RegAddr = CGF.Builder.CreateBitCast(BaseAddr, MemTy);
  }

  CGF.EmitBranch(ContBlock);

  //=======================================
  // Argument was on the stack
  //=======================================
  CGF.EmitBlock(OnStackBlock);

  llvm::Value *stack_p = nullptr, *OnStackAddr = nullptr;
  stack_p = CGF.Builder.CreateStructGEP(VAListAddr, 0, "stack_p");
  OnStackAddr = CGF.Builder.CreateLoad(stack_p, "stack");

  // Again, stack arguments may need realigmnent. In this case both integer and
  // floating-point ones might be affected.
  if (!IsIndirect && Ctx.getTypeAlign(Ty) > 64) {
    int Align = Ctx.getTypeAlign(Ty) / 8;

    OnStackAddr = CGF.Builder.CreatePtrToInt(OnStackAddr, CGF.Int64Ty);

    OnStackAddr = CGF.Builder.CreateAdd(
        OnStackAddr, llvm::ConstantInt::get(CGF.Int64Ty, Align - 1),
        "align_stack");
    OnStackAddr = CGF.Builder.CreateAnd(
        OnStackAddr, llvm::ConstantInt::get(CGF.Int64Ty, -Align),
        "align_stack");

    OnStackAddr = CGF.Builder.CreateIntToPtr(OnStackAddr, CGF.Int8PtrTy);
  }

  uint64_t StackSize;
  if (IsIndirect)
    StackSize = 8;
  else
    StackSize = Ctx.getTypeSize(Ty) / 8;

  // All stack slots are 8 bytes
  StackSize = llvm::RoundUpToAlignment(StackSize, 8);

  llvm::Value *StackSizeC = llvm::ConstantInt::get(CGF.Int32Ty, StackSize);
  llvm::Value *NewStack =
      CGF.Builder.CreateGEP(OnStackAddr, StackSizeC, "new_stack");

  // Write the new value of __stack for the next call to va_arg
  CGF.Builder.CreateStore(NewStack, stack_p);

  if (CGF.CGM.getDataLayout().isBigEndian() && !isAggregateTypeForABI(Ty) &&
      Ctx.getTypeSize(Ty) < 64) {
    int Offset = 8 - Ctx.getTypeSize(Ty) / 8;
    OnStackAddr = CGF.Builder.CreatePtrToInt(OnStackAddr, CGF.Int64Ty);

    OnStackAddr = CGF.Builder.CreateAdd(
        OnStackAddr, llvm::ConstantInt::get(CGF.Int64Ty, Offset), "align_be");

    OnStackAddr = CGF.Builder.CreateIntToPtr(OnStackAddr, CGF.Int8PtrTy);
  }

  OnStackAddr = CGF.Builder.CreateBitCast(OnStackAddr, MemTy);

  CGF.EmitBranch(ContBlock);

  //=======================================
  // Tidy up
  //=======================================
  CGF.EmitBlock(ContBlock);

  llvm::PHINode *ResAddr = CGF.Builder.CreatePHI(MemTy, 2, "vaarg.addr");
  ResAddr->addIncoming(RegAddr, InRegBlock);
  ResAddr->addIncoming(OnStackAddr, OnStackBlock);

  if (IsIndirect)
    return CGF.Builder.CreateLoad(ResAddr, "vaarg.addr");

  return ResAddr;
}

llvm::Value *AArch64ABIInfo::EmitDarwinVAArg(llvm::Value *VAListAddr, QualType Ty,
                                           CodeGenFunction &CGF) const {
  // We do not support va_arg for aggregates or illegal vector types.
  // Lower VAArg here for these cases and use the LLVM va_arg instruction for
  // other cases.
  if (!isAggregateTypeForABI(Ty) && !isIllegalVectorType(Ty))
    return nullptr;

  uint64_t Size = CGF.getContext().getTypeSize(Ty) / 8;
  uint64_t Align = CGF.getContext().getTypeAlign(Ty) / 8;

  const Type *Base = nullptr;
  uint64_t Members = 0;
  bool isHA = isHomogeneousAggregate(Ty, Base, Members);

  bool isIndirect = false;
  // Arguments bigger than 16 bytes which aren't homogeneous aggregates should
  // be passed indirectly.
  if (Size > 16 && !isHA) {
    isIndirect = true;
    Size = 8;
    Align = 8;
  }

  llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  llvm::Type *BPP = llvm::PointerType::getUnqual(BP);

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");

  if (isEmptyRecord(getContext(), Ty, true)) {
    // These are ignored for parameter passing purposes.
    llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
    return Builder.CreateBitCast(Addr, PTy);
  }

  const uint64_t MinABIAlign = 8;
  if (Align > MinABIAlign) {
    llvm::Value *Offset = llvm::ConstantInt::get(CGF.Int32Ty, Align - 1);
    Addr = Builder.CreateGEP(Addr, Offset);
    llvm::Value *AsInt = Builder.CreatePtrToInt(Addr, CGF.Int64Ty);
    llvm::Value *Mask = llvm::ConstantInt::get(CGF.Int64Ty, ~(Align - 1));
    llvm::Value *Aligned = Builder.CreateAnd(AsInt, Mask);
    Addr = Builder.CreateIntToPtr(Aligned, BP, "ap.align");
  }

  uint64_t Offset = llvm::RoundUpToAlignment(Size, MinABIAlign);
  llvm::Value *NextAddr = Builder.CreateGEP(
      Addr, llvm::ConstantInt::get(CGF.Int32Ty, Offset), "ap.next");
  Builder.CreateStore(NextAddr, VAListAddrAsBPP);

  if (isIndirect)
    Addr = Builder.CreateLoad(Builder.CreateBitCast(Addr, BPP));
  llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped = Builder.CreateBitCast(Addr, PTy);

  return AddrTyped;
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
  mutable int VFPRegs[16];
  const unsigned NumVFPs;
  const unsigned NumGPRs;
  mutable unsigned AllocatedGPRs;
  mutable unsigned AllocatedVFPs;

public:
  ARMABIInfo(CodeGenTypes &CGT, ABIKind _Kind) : ABIInfo(CGT), Kind(_Kind),
    NumVFPs(16), NumGPRs(4) {
    setCCs();
    resetAllocatedRegs();
  }

  bool isEABI() const {
    switch (getTarget().getTriple().getEnvironment()) {
    case llvm::Triple::Android:
    case llvm::Triple::EABI:
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::GNUEABIHF:
      return true;
    default:
      return false;
    }
  }

  bool isEABIHF() const {
    switch (getTarget().getTriple().getEnvironment()) {
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABIHF:
      return true;
    default:
      return false;
    }
  }

  ABIKind getABIKind() const { return Kind; }

private:
  ABIArgInfo classifyReturnType(QualType RetTy, bool isVariadic) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, bool isVariadic,
                                  bool &IsCPRC) const;
  bool isIllegalVectorType(QualType Ty) const;

  bool isHomogeneousAggregateBaseType(QualType Ty) const override;
  bool isHomogeneousAggregateSmallEnough(const Type *Ty,
                                         uint64_t Members) const override;

  void computeInfo(CGFunctionInfo &FI) const override;

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;

  llvm::CallingConv::ID getLLVMDefaultCC() const;
  llvm::CallingConv::ID getABIDefaultCC() const;
  void setCCs();

  void markAllocatedGPRs(unsigned Alignment, unsigned NumRequired) const;
  void markAllocatedVFPs(unsigned Alignment, unsigned NumRequired) const;
  void resetAllocatedRegs(void) const;
};

class ARMTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  ARMTargetCodeGenInfo(CodeGenTypes &CGT, ARMABIInfo::ABIKind K)
    :TargetCodeGenInfo(new ARMABIInfo(CGT, K)) {}

  const ARMABIInfo &getABIInfo() const {
    return static_cast<const ARMABIInfo&>(TargetCodeGenInfo::getABIInfo());
  }

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    return 13;
  }

  StringRef getARCRetainAutoreleasedReturnValueMarker() const override {
    return "mov\tr7, r7\t\t@ marker for objc_retainAutoreleaseReturnValue";
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override {
    llvm::Value *Four8 = llvm::ConstantInt::get(CGF.Int8Ty, 4);

    // 0-15 are the 16 integer registers.
    AssignToArrayRange(CGF.Builder, Address, Four8, 0, 15);
    return false;
  }

  unsigned getSizeOfUnwindException() const override {
    if (getABIInfo().isEABI()) return 88;
    return TargetCodeGenInfo::getSizeOfUnwindException();
  }

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override {
    const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
    if (!FD)
      return;

    const ARMInterruptAttr *Attr = FD->getAttr<ARMInterruptAttr>();
    if (!Attr)
      return;

    const char *Kind;
    switch (Attr->getInterrupt()) {
    case ARMInterruptAttr::Generic: Kind = ""; break;
    case ARMInterruptAttr::IRQ:     Kind = "IRQ"; break;
    case ARMInterruptAttr::FIQ:     Kind = "FIQ"; break;
    case ARMInterruptAttr::SWI:     Kind = "SWI"; break;
    case ARMInterruptAttr::ABORT:   Kind = "ABORT"; break;
    case ARMInterruptAttr::UNDEF:   Kind = "UNDEF"; break;
    }

    llvm::Function *Fn = cast<llvm::Function>(GV);

    Fn->addFnAttr("interrupt", Kind);

    if (cast<ARMABIInfo>(getABIInfo()).getABIKind() == ARMABIInfo::APCS)
      return;

    // AAPCS guarantees that sp will be 8-byte aligned on any public interface,
    // however this is not necessarily true on taking any interrupt. Instruct
    // the backend to perform a realignment as part of the function prologue.
    llvm::AttrBuilder B;
    B.addStackAlignmentAttr(8);
    Fn->addAttributes(llvm::AttributeSet::FunctionIndex,
                      llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                              B));
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
  resetAllocatedRegs();

  if (getCXXABI().classifyReturnType(FI)) {
    if (FI.getReturnInfo().isIndirect())
      markAllocatedGPRs(1, 1);
  } else {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), FI.isVariadic());
  }
  for (auto &I : FI.arguments()) {
    unsigned PreAllocationVFPs = AllocatedVFPs;
    unsigned PreAllocationGPRs = AllocatedGPRs;
    bool IsCPRC = false;
    // 6.1.2.3 There is one VFP co-processor register class using registers
    // s0-s15 (d0-d7) for passing arguments.
    I.info = classifyArgumentType(I.type, FI.isVariadic(), IsCPRC);

    // If we have allocated some arguments onto the stack (due to running
    // out of VFP registers), we cannot split an argument between GPRs and
    // the stack. If this situation occurs, we add padding to prevent the
    // GPRs from being used. In this situation, the current argument could
    // only be allocated by rule C.8, so rule C.6 would mark these GPRs as
    // unusable anyway.
    // We do not have to do this if the argument is being passed ByVal, as the
    // backend can handle that situation correctly.
    const bool StackUsed = PreAllocationGPRs > NumGPRs || PreAllocationVFPs > NumVFPs;
    const bool IsByVal = I.info.isIndirect() && I.info.getIndirectByVal();
    if (!IsCPRC && PreAllocationGPRs < NumGPRs && AllocatedGPRs > NumGPRs &&
        StackUsed && !IsByVal) {
      llvm::Type *PaddingTy = llvm::ArrayType::get(
          llvm::Type::getInt32Ty(getVMContext()), NumGPRs - PreAllocationGPRs);
      if (I.info.canHaveCoerceToType()) {
        I.info = ABIArgInfo::getDirect(I.info.getCoerceToType() /* type */,
                                       0 /* offset */, PaddingTy, true);
      } else {
        I.info = ABIArgInfo::getDirect(nullptr /* type */, 0 /* offset */,
                                       PaddingTy, true);
      }
    }
  }

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  llvm::CallingConv::ID cc = getRuntimeCC();
  if (cc != llvm::CallingConv::C)
    FI.setEffectiveCallingConvention(cc);    
}

/// Return the default calling convention that LLVM will use.
llvm::CallingConv::ID ARMABIInfo::getLLVMDefaultCC() const {
  // The default calling convention that LLVM will infer.
  if (isEABIHF())
    return llvm::CallingConv::ARM_AAPCS_VFP;
  else if (isEABI())
    return llvm::CallingConv::ARM_AAPCS;
  else
    return llvm::CallingConv::ARM_APCS;
}

/// Return the calling convention that our ABI would like us to use
/// as the C calling convention.
llvm::CallingConv::ID ARMABIInfo::getABIDefaultCC() const {
  switch (getABIKind()) {
  case APCS: return llvm::CallingConv::ARM_APCS;
  case AAPCS: return llvm::CallingConv::ARM_AAPCS;
  case AAPCS_VFP: return llvm::CallingConv::ARM_AAPCS_VFP;
  }
  llvm_unreachable("bad ABI kind");
}

void ARMABIInfo::setCCs() {
  assert(getRuntimeCC() == llvm::CallingConv::C);

  // Don't muddy up the IR with a ton of explicit annotations if
  // they'd just match what LLVM will infer from the triple.
  llvm::CallingConv::ID abiCC = getABIDefaultCC();
  if (abiCC != getLLVMDefaultCC())
    RuntimeCC = abiCC;

  BuiltinCC = (getABIKind() == APCS ?
               llvm::CallingConv::ARM_APCS : llvm::CallingConv::ARM_AAPCS);
}

/// markAllocatedVFPs - update VFPRegs according to the alignment and
/// number of VFP registers (unit is S register) requested.
void ARMABIInfo::markAllocatedVFPs(unsigned Alignment,
                                   unsigned NumRequired) const {
  // Early Exit.
  if (AllocatedVFPs >= 16) {
    // We use AllocatedVFP > 16 to signal that some CPRCs were allocated on
    // the stack.
    AllocatedVFPs = 17;
    return;
  }
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
      AllocatedVFPs += NumRequired;
      return;
    }
  }
  // C.2.vfp If the argument is a VFP CPRC then any VFP registers that are
  // unallocated are marked as unavailable.
  for (unsigned I = 0; I < 16; I++)
    VFPRegs[I] = 1;
  AllocatedVFPs = 17; // We do not have enough VFP registers.
}

/// Update AllocatedGPRs to record the number of general purpose registers
/// which have been allocated. It is valid for AllocatedGPRs to go above 4,
/// this represents arguments being stored on the stack.
void ARMABIInfo::markAllocatedGPRs(unsigned Alignment,
                                   unsigned NumRequired) const {
  assert((Alignment == 1 || Alignment == 2) && "Alignment must be 4 or 8 bytes");

  if (Alignment == 2 && AllocatedGPRs & 0x1)
    AllocatedGPRs += 1;

  AllocatedGPRs += NumRequired;
}

void ARMABIInfo::resetAllocatedRegs(void) const {
  AllocatedGPRs = 0;
  AllocatedVFPs = 0;
  for (unsigned i = 0; i < NumVFPs; ++i)
    VFPRegs[i] = 0;
}

ABIArgInfo ARMABIInfo::classifyArgumentType(QualType Ty, bool isVariadic,
                                            bool &IsCPRC) const {
  // We update number of allocated VFPs according to
  // 6.1.2.1 The following argument types are VFP CPRCs:
  //   A single-precision floating-point type (including promoted
  //   half-precision types); A double-precision floating-point type;
  //   A 64-bit or 128-bit containerized vector type; Homogeneous Aggregate
  //   with a Base Type of a single- or double-precision floating-point type,
  //   64-bit containerized vectors or 128-bit containerized vectors with one
  //   to four Elements.
  bool IsEffectivelyAAPCS_VFP = getABIKind() == AAPCS_VFP && !isVariadic;

  Ty = useFirstFieldIfTransparentUnion(Ty);

  // Handle illegal vector types here.
  if (isIllegalVectorType(Ty)) {
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size <= 32) {
      llvm::Type *ResType =
          llvm::Type::getInt32Ty(getVMContext());
      markAllocatedGPRs(1, 1);
      return ABIArgInfo::getDirect(ResType);
    }
    if (Size == 64) {
      llvm::Type *ResType = llvm::VectorType::get(
          llvm::Type::getInt32Ty(getVMContext()), 2);
      if (getABIKind() == ARMABIInfo::AAPCS || isVariadic){
        markAllocatedGPRs(2, 2);
      } else {
        markAllocatedVFPs(2, 2);
        IsCPRC = true;
      }
      return ABIArgInfo::getDirect(ResType);
    }
    if (Size == 128) {
      llvm::Type *ResType = llvm::VectorType::get(
          llvm::Type::getInt32Ty(getVMContext()), 4);
      if (getABIKind() == ARMABIInfo::AAPCS || isVariadic) {
        markAllocatedGPRs(2, 4);
      } else {
        markAllocatedVFPs(4, 4);
        IsCPRC = true;
      }
      return ABIArgInfo::getDirect(ResType);
    }
    markAllocatedGPRs(1, 1);
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);
  }
  // Update VFPRegs for legal vector types.
  if (getABIKind() == ARMABIInfo::AAPCS_VFP && !isVariadic) {
    if (const VectorType *VT = Ty->getAs<VectorType>()) {
      uint64_t Size = getContext().getTypeSize(VT);
      // Size of a legal vector should be power of 2 and above 64.
      markAllocatedVFPs(Size >= 128 ? 4 : 2, Size / 32);
      IsCPRC = true;
    }
  }
  // Update VFPRegs for floating point types.
  if (getABIKind() == ARMABIInfo::AAPCS_VFP && !isVariadic) {
    if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
      if (BT->getKind() == BuiltinType::Half ||
          BT->getKind() == BuiltinType::Float) {
        markAllocatedVFPs(1, 1);
        IsCPRC = true;
      }
      if (BT->getKind() == BuiltinType::Double ||
          BT->getKind() == BuiltinType::LongDouble) {
        markAllocatedVFPs(2, 2);
        IsCPRC = true;
      }
    }
  }

  if (!isAggregateTypeForABI(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>()) {
      Ty = EnumTy->getDecl()->getIntegerType();
    }

    unsigned Size = getContext().getTypeSize(Ty);
    if (!IsCPRC)
      markAllocatedGPRs(Size > 32 ? 2 : 1, (Size + 31) / 32);
    return (Ty->isPromotableIntegerType() ? ABIArgInfo::getExtend()
                                          : ABIArgInfo::getDirect());
  }

  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI())) {
    markAllocatedGPRs(1, 1);
    return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);
  }

  // Ignore empty records.
  if (isEmptyRecord(getContext(), Ty, true))
    return ABIArgInfo::getIgnore();

  if (IsEffectivelyAAPCS_VFP) {
    // Homogeneous Aggregates need to be expanded when we can fit the aggregate
    // into VFP registers.
    const Type *Base = nullptr;
    uint64_t Members = 0;
    if (isHomogeneousAggregate(Ty, Base, Members)) {
      assert(Base && "Base class should be set for homogeneous aggregate");
      // Base can be a floating-point or a vector.
      if (Base->isVectorType()) {
        // ElementSize is in number of floats.
        unsigned ElementSize = getContext().getTypeSize(Base) == 64 ? 2 : 4;
        markAllocatedVFPs(ElementSize,
                          Members * ElementSize);
      } else if (Base->isSpecificBuiltinType(BuiltinType::Float))
        markAllocatedVFPs(1, Members);
      else {
        assert(Base->isSpecificBuiltinType(BuiltinType::Double) ||
               Base->isSpecificBuiltinType(BuiltinType::LongDouble));
        markAllocatedVFPs(2, Members * 2);
      }
      IsCPRC = true;
      return ABIArgInfo::getDirect(nullptr, 0, nullptr, false);
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
    // Update Allocated GPRs. Since this is only used when the size of the
    // argument is greater than 64 bytes, this will always use up any available
    // registers (of which there are 4). We also don't care about getting the
    // alignment right, because general-purpose registers cannot be back-filled.
    markAllocatedGPRs(1, 4);
    return ABIArgInfo::getIndirect(TyAlign, /*ByVal=*/true,
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
    markAllocatedGPRs(1, SizeRegs);
  } else {
    ElemTy = llvm::Type::getInt64Ty(getVMContext());
    SizeRegs = (getContext().getTypeSize(Ty) + 63) / 64;
    markAllocatedGPRs(2, SizeRegs * 2);
  }

  return ABIArgInfo::getDirect(llvm::ArrayType::get(ElemTy, SizeRegs));
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

ABIArgInfo ARMABIInfo::classifyReturnType(QualType RetTy,
                                          bool isVariadic) const {
  bool IsEffectivelyAAPCS_VFP = getABIKind() == AAPCS_VFP && !isVariadic;

  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // Large vector types should be returned via memory.
  if (RetTy->isVectorType() && getContext().getTypeSize(RetTy) > 128) {
    markAllocatedGPRs(1, 1);
    return ABIArgInfo::getIndirect(0);
  }

  if (!isAggregateTypeForABI(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    return RetTy->isPromotableIntegerType() ? ABIArgInfo::getExtend()
                                            : ABIArgInfo::getDirect();
  }

  // Are we following APCS?
  if (getABIKind() == APCS) {
    if (isEmptyRecord(getContext(), RetTy, false))
      return ABIArgInfo::getIgnore();

    // Complex types are all returned as packed integers.
    //
    // FIXME: Consider using 2 x vector types if the back end handles them
    // correctly.
    if (RetTy->isAnyComplexType())
      return ABIArgInfo::getDirect(llvm::IntegerType::get(
          getVMContext(), getContext().getTypeSize(RetTy)));

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
    markAllocatedGPRs(1, 1);
    return ABIArgInfo::getIndirect(0);
  }

  // Otherwise this is an AAPCS variant.

  if (isEmptyRecord(getContext(), RetTy, true))
    return ABIArgInfo::getIgnore();

  // Check for homogeneous aggregates with AAPCS-VFP.
  if (IsEffectivelyAAPCS_VFP) {
    const Type *Base = nullptr;
    uint64_t Members;
    if (isHomogeneousAggregate(RetTy, Base, Members)) {
      assert(Base && "Base class should be set for homogeneous aggregate");
      // Homogeneous Aggregates are returned directly.
      return ABIArgInfo::getDirect(nullptr, 0, nullptr, false);
    }
  }

  // Aggregates <= 4 bytes are returned in r0; other aggregates
  // are returned indirectly.
  uint64_t Size = getContext().getTypeSize(RetTy);
  if (Size <= 32) {
    if (getDataLayout().isBigEndian())
      // Return in 32 bit integer integer type (as if loaded by LDR, AAPCS 5.4)
      return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

    // Return in the smallest viable integer type.
    if (Size <= 8)
      return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
    if (Size <= 16)
      return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));
    return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));
  }

  markAllocatedGPRs(1, 1);
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

bool ARMABIInfo::isHomogeneousAggregateBaseType(QualType Ty) const {
  // Homogeneous aggregates for AAPCS-VFP must have base types of float,
  // double, or 64-bit or 128-bit vectors.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    if (BT->getKind() == BuiltinType::Float ||
        BT->getKind() == BuiltinType::Double ||
        BT->getKind() == BuiltinType::LongDouble)
      return true;
  } else if (const VectorType *VT = Ty->getAs<VectorType>()) {
    unsigned VecSize = getContext().getTypeSize(VT);
    if (VecSize == 64 || VecSize == 128)
      return true;
  }
  return false;
}

bool ARMABIInfo::isHomogeneousAggregateSmallEnough(const Type *Base,
                                                   uint64_t Members) const {
  return Members <= 4;
}

llvm::Value *ARMABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                   CodeGenFunction &CGF) const {
  llvm::Type *BP = CGF.Int8PtrTy;
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");

  if (isEmptyRecord(getContext(), Ty, true)) {
    // These are ignored for parameter passing purposes.
    llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
    return Builder.CreateBitCast(Addr, PTy);
  }

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
  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
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

  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CFG) const override;
};

class NVPTXTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  NVPTXTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new NVPTXABIInfo(CGT)) {}

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const override;
private:
  // Adds a NamedMDNode with F, Name, and Operand as operands, and adds the
  // resulting MDNode to the nvvm.annotations MDNode.
  static void addNVVMMetadata(llvm::Function *F, StringRef Name, int Operand);
};

ABIArgInfo NVPTXABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // note: this is different from default ABI
  if (!RetTy->isScalarType())
    return ABIArgInfo::getDirect();

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (RetTy->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo NVPTXABIInfo::classifyArgumentType(QualType Ty) const {
  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  // Return aggregates type as indirect by value
  if (isAggregateTypeForABI(Ty))
    return ABIArgInfo::getIndirect(0, /* byval */ true);

  return (Ty->isPromotableIntegerType() ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

void NVPTXABIInfo::computeInfo(CGFunctionInfo &FI) const {
  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (auto &I : FI.arguments())
    I.info = classifyArgumentType(I.type);

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  FI.setEffectiveCallingConvention(getRuntimeCC());
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
    // Use OpenCL function attributes to check for kernel functions
    // By default, all functions are device functions
    if (FD->hasAttr<OpenCLKernelAttr>()) {
      // OpenCL __kernel functions get kernel metadata
      // Create !{<func-ref>, metadata !"kernel", i32 1} node
      addNVVMMetadata(F, "kernel", 1);
      // And kernel functions are not subject to inlining
      F->addFnAttr(llvm::Attribute::NoInline);
    }
  }

  // Perform special handling in CUDA mode.
  if (M.getLangOpts().CUDA) {
    // CUDA __global__ functions get a kernel metadata entry.  Since
    // __global__ functions cannot be called from the device, we do not
    // need to set the noinline attribute.
    if (FD->hasAttr<CUDAGlobalAttr>()) {
      // Create !{<func-ref>, metadata !"kernel", i32 1} node
      addNVVMMetadata(F, "kernel", 1);
    }
    if (FD->hasAttr<CUDALaunchBoundsAttr>()) {
      // Create !{<func-ref>, metadata !"maxntidx", i32 <val>} node
      addNVVMMetadata(F, "maxntidx",
                      FD->getAttr<CUDALaunchBoundsAttr>()->getMaxThreads());
      // min blocks is a default argument for CUDALaunchBoundsAttr, so getting a
      // zero value from getMinBlocks either means it was not specified in
      // __launch_bounds__ or the user specified a 0 value. In both cases, we
      // don't have to add a PTX directive.
      int MinCTASM = FD->getAttr<CUDALaunchBoundsAttr>()->getMinBlocks();
      if (MinCTASM > 0) {
        // Create !{<func-ref>, metadata !"minctasm", i32 <val>} node
        addNVVMMetadata(F, "minctasm", MinCTASM);
      }
    }
  }
}

void NVPTXTargetCodeGenInfo::addNVVMMetadata(llvm::Function *F, StringRef Name,
                                             int Operand) {
  llvm::Module *M = F->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(F), llvm::MDString::get(Ctx, Name),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Operand))};
  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}
}

//===----------------------------------------------------------------------===//
// SystemZ ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class SystemZABIInfo : public ABIInfo {
public:
  SystemZABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

  bool isPromotableIntegerType(QualType Ty) const;
  bool isCompoundType(QualType Ty) const;
  bool isFPArgumentType(QualType Ty) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType ArgTy) const;

  void computeInfo(CGFunctionInfo &FI) const override {
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &I : FI.arguments())
      I.info = classifyArgumentType(I.type);
  }

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class SystemZTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  SystemZTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new SystemZABIInfo(CGT)) {}
};

}

bool SystemZABIInfo::isPromotableIntegerType(QualType Ty) const {
  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  // Promotable integer types are required to be promoted by the ABI.
  if (Ty->isPromotableIntegerType())
    return true;

  // 32-bit values must also be promoted.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return true;
    default:
      return false;
    }
  return false;
}

bool SystemZABIInfo::isCompoundType(QualType Ty) const {
  return Ty->isAnyComplexType() || isAggregateTypeForABI(Ty);
}

bool SystemZABIInfo::isFPArgumentType(QualType Ty) const {
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Float:
    case BuiltinType::Double:
      return true;
    default:
      return false;
    }

  if (const RecordType *RT = Ty->getAsStructureType()) {
    const RecordDecl *RD = RT->getDecl();
    bool Found = false;

    // If this is a C++ record, check the bases first.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
      for (const auto &I : CXXRD->bases()) {
        QualType Base = I.getType();

        // Empty bases don't affect things either way.
        if (isEmptyRecord(getContext(), Base, true))
          continue;

        if (Found)
          return false;
        Found = isFPArgumentType(Base);
        if (!Found)
          return false;
      }

    // Check the fields.
    for (const auto *FD : RD->fields()) {
      // Empty bitfields don't affect things either way.
      // Unlike isSingleElementStruct(), empty structure and array fields
      // do count.  So do anonymous bitfields that aren't zero-sized.
      if (FD->isBitField() && FD->getBitWidthValue(getContext()) == 0)
        return true;

      // Unlike isSingleElementStruct(), arrays do not count.
      // Nested isFPArgumentType structures still do though.
      if (Found)
        return false;
      Found = isFPArgumentType(FD->getType());
      if (!Found)
        return false;
    }

    // Unlike isSingleElementStruct(), trailing padding is allowed.
    // An 8-byte aligned struct s { float f; } is passed as a double.
    return Found;
  }

  return false;
}

llvm::Value *SystemZABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  // Assume that va_list type is correct; should be pointer to LLVM type:
  // struct {
  //   i64 __gpr;
  //   i64 __fpr;
  //   i8 *__overflow_arg_area;
  //   i8 *__reg_save_area;
  // };

  // Every argument occupies 8 bytes and is passed by preference in either
  // GPRs or FPRs.
  Ty = CGF.getContext().getCanonicalType(Ty);
  ABIArgInfo AI = classifyArgumentType(Ty);
  bool InFPRs = isFPArgumentType(Ty);

  llvm::Type *APTy = llvm::PointerType::getUnqual(CGF.ConvertTypeForMem(Ty));
  bool IsIndirect = AI.isIndirect();
  unsigned UnpaddedBitSize;
  if (IsIndirect) {
    APTy = llvm::PointerType::getUnqual(APTy);
    UnpaddedBitSize = 64;
  } else
    UnpaddedBitSize = getContext().getTypeSize(Ty);
  unsigned PaddedBitSize = 64;
  assert((UnpaddedBitSize <= PaddedBitSize) && "Invalid argument size.");

  unsigned PaddedSize = PaddedBitSize / 8;
  unsigned Padding = (PaddedBitSize - UnpaddedBitSize) / 8;

  unsigned MaxRegs, RegCountField, RegSaveIndex, RegPadding;
  if (InFPRs) {
    MaxRegs = 4; // Maximum of 4 FPR arguments
    RegCountField = 1; // __fpr
    RegSaveIndex = 16; // save offset for f0
    RegPadding = 0; // floats are passed in the high bits of an FPR
  } else {
    MaxRegs = 5; // Maximum of 5 GPR arguments
    RegCountField = 0; // __gpr
    RegSaveIndex = 2; // save offset for r2
    RegPadding = Padding; // values are passed in the low bits of a GPR
  }

  llvm::Value *RegCountPtr =
    CGF.Builder.CreateStructGEP(VAListAddr, RegCountField, "reg_count_ptr");
  llvm::Value *RegCount = CGF.Builder.CreateLoad(RegCountPtr, "reg_count");
  llvm::Type *IndexTy = RegCount->getType();
  llvm::Value *MaxRegsV = llvm::ConstantInt::get(IndexTy, MaxRegs);
  llvm::Value *InRegs = CGF.Builder.CreateICmpULT(RegCount, MaxRegsV,
                                                 "fits_in_regs");

  llvm::BasicBlock *InRegBlock = CGF.createBasicBlock("vaarg.in_reg");
  llvm::BasicBlock *InMemBlock = CGF.createBasicBlock("vaarg.in_mem");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("vaarg.end");
  CGF.Builder.CreateCondBr(InRegs, InRegBlock, InMemBlock);

  // Emit code to load the value if it was passed in registers.
  CGF.EmitBlock(InRegBlock);

  // Work out the address of an argument register.
  llvm::Value *PaddedSizeV = llvm::ConstantInt::get(IndexTy, PaddedSize);
  llvm::Value *ScaledRegCount =
    CGF.Builder.CreateMul(RegCount, PaddedSizeV, "scaled_reg_count");
  llvm::Value *RegBase =
    llvm::ConstantInt::get(IndexTy, RegSaveIndex * PaddedSize + RegPadding);
  llvm::Value *RegOffset =
    CGF.Builder.CreateAdd(ScaledRegCount, RegBase, "reg_offset");
  llvm::Value *RegSaveAreaPtr =
    CGF.Builder.CreateStructGEP(VAListAddr, 3, "reg_save_area_ptr");
  llvm::Value *RegSaveArea =
    CGF.Builder.CreateLoad(RegSaveAreaPtr, "reg_save_area");
  llvm::Value *RawRegAddr =
    CGF.Builder.CreateGEP(RegSaveArea, RegOffset, "raw_reg_addr");
  llvm::Value *RegAddr =
    CGF.Builder.CreateBitCast(RawRegAddr, APTy, "reg_addr");

  // Update the register count
  llvm::Value *One = llvm::ConstantInt::get(IndexTy, 1);
  llvm::Value *NewRegCount =
    CGF.Builder.CreateAdd(RegCount, One, "reg_count");
  CGF.Builder.CreateStore(NewRegCount, RegCountPtr);
  CGF.EmitBranch(ContBlock);

  // Emit code to load the value if it was passed in memory.
  CGF.EmitBlock(InMemBlock);

  // Work out the address of a stack argument.
  llvm::Value *OverflowArgAreaPtr =
    CGF.Builder.CreateStructGEP(VAListAddr, 2, "overflow_arg_area_ptr");
  llvm::Value *OverflowArgArea =
    CGF.Builder.CreateLoad(OverflowArgAreaPtr, "overflow_arg_area");
  llvm::Value *PaddingV = llvm::ConstantInt::get(IndexTy, Padding);
  llvm::Value *RawMemAddr =
    CGF.Builder.CreateGEP(OverflowArgArea, PaddingV, "raw_mem_addr");
  llvm::Value *MemAddr =
    CGF.Builder.CreateBitCast(RawMemAddr, APTy, "mem_addr");

  // Update overflow_arg_area_ptr pointer
  llvm::Value *NewOverflowArgArea =
    CGF.Builder.CreateGEP(OverflowArgArea, PaddedSizeV, "overflow_arg_area");
  CGF.Builder.CreateStore(NewOverflowArgArea, OverflowArgAreaPtr);
  CGF.EmitBranch(ContBlock);

  // Return the appropriate result.
  CGF.EmitBlock(ContBlock);
  llvm::PHINode *ResAddr = CGF.Builder.CreatePHI(APTy, 2, "va_arg.addr");
  ResAddr->addIncoming(RegAddr, InRegBlock);
  ResAddr->addIncoming(MemAddr, InMemBlock);

  if (IsIndirect)
    return CGF.Builder.CreateLoad(ResAddr, "indirect_arg");

  return ResAddr;
}

ABIArgInfo SystemZABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();
  if (isCompoundType(RetTy) || getContext().getTypeSize(RetTy) > 64)
    return ABIArgInfo::getIndirect(0);
  return (isPromotableIntegerType(RetTy) ?
          ABIArgInfo::getExtend() : ABIArgInfo::getDirect());
}

ABIArgInfo SystemZABIInfo::classifyArgumentType(QualType Ty) const {
  // Handle the generic C++ ABI.
  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
    return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);

  // Integers and enums are extended to full register width.
  if (isPromotableIntegerType(Ty))
    return ABIArgInfo::getExtend();

  // Values that are not 1, 2, 4 or 8 bytes in size are passed indirectly.
  uint64_t Size = getContext().getTypeSize(Ty);
  if (Size != 8 && Size != 16 && Size != 32 && Size != 64)
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Handle small structures.
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    // Structures with flexible arrays have variable length, so really
    // fail the size test above.
    const RecordDecl *RD = RT->getDecl();
    if (RD->hasFlexibleArrayMember())
      return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

    // The structure is passed as an unextended integer, a float, or a double.
    llvm::Type *PassTy;
    if (isFPArgumentType(Ty)) {
      assert(Size == 32 || Size == 64);
      if (Size == 32)
        PassTy = llvm::Type::getFloatTy(getVMContext());
      else
        PassTy = llvm::Type::getDoubleTy(getVMContext());
    } else
      PassTy = llvm::IntegerType::get(getVMContext(), Size);
    return ABIArgInfo::getDirect(PassTy);
  }

  // Non-structure compounds are passed indirectly.
  if (isCompoundType(Ty))
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  return ABIArgInfo::getDirect(nullptr);
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
                           CodeGen::CodeGenModule &M) const override;
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
      unsigned Num = attr->getNumber() / 2;
      llvm::GlobalAlias::create(llvm::Function::ExternalLinkage,
                                "__isr_" + Twine(Num), F);
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
                       SmallVectorImpl<llvm::Type *> &ArgList) const;
  llvm::Type* HandleAggregates(QualType Ty, uint64_t TySize) const;
  llvm::Type* returnAggregateInRegs(QualType RetTy, uint64_t Size) const;
  llvm::Type* getPaddingType(uint64_t Align, uint64_t Offset) const;
public:
  MipsABIInfo(CodeGenTypes &CGT, bool _IsO32) :
    ABIInfo(CGT), IsO32(_IsO32), MinABIStackAlignInBytes(IsO32 ? 4 : 8),
    StackAlignInBytes(IsO32 ? 8 : 16) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, uint64_t &Offset) const;
  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class MIPSTargetCodeGenInfo : public TargetCodeGenInfo {
  unsigned SizeOfUnwindException;
public:
  MIPSTargetCodeGenInfo(CodeGenTypes &CGT, bool IsO32)
    : TargetCodeGenInfo(new MipsABIInfo(CGT, IsO32)),
      SizeOfUnwindException(IsO32 ? 24 : 32) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &CGM) const override {
    return 29;
  }

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override {
    const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
    if (!FD) return;
    llvm::Function *Fn = cast<llvm::Function>(GV);
    if (FD->hasAttr<Mips16Attr>()) {
      Fn->addFnAttr("mips16");
    }
    else if (FD->hasAttr<NoMips16Attr>()) {
      Fn->addFnAttr("nomips16");
    }
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override;

  unsigned getSizeOfUnwindException() const override {
    return SizeOfUnwindException;
  }
};
}

void MipsABIInfo::CoerceToIntArgs(uint64_t TySize,
                                  SmallVectorImpl<llvm::Type *> &ArgList) const {
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

llvm::Type *MipsABIInfo::getPaddingType(uint64_t OrigOffset,
                                        uint64_t Offset) const {
  if (OrigOffset + MinABIStackAlignInBytes > Offset)
    return nullptr;

  return llvm::IntegerType::get(getVMContext(), (Offset - OrigOffset) * 8);
}

ABIArgInfo
MipsABIInfo::classifyArgumentType(QualType Ty, uint64_t &Offset) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  uint64_t OrigOffset = Offset;
  uint64_t TySize = getContext().getTypeSize(Ty);
  uint64_t Align = getContext().getTypeAlign(Ty) / 8;

  Align = std::min(std::max(Align, (uint64_t)MinABIStackAlignInBytes),
                   (uint64_t)StackAlignInBytes);
  unsigned CurrOffset = llvm::RoundUpToAlignment(Offset, Align);
  Offset = CurrOffset + llvm::RoundUpToAlignment(TySize, Align * 8) / 8;

  if (isAggregateTypeForABI(Ty) || Ty->isVectorType()) {
    // Ignore empty aggregates.
    if (TySize == 0)
      return ABIArgInfo::getIgnore();

    if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI())) {
      Offset = OrigOffset + MinABIStackAlignInBytes;
      return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);
    }

    // If we have reached here, aggregates are passed directly by coercing to
    // another structure type. Padding is inserted if the offset of the
    // aggregate is unaligned.
    ABIArgInfo ArgInfo =
        ABIArgInfo::getDirect(HandleAggregates(Ty, TySize), 0,
                              getPaddingType(OrigOffset, CurrOffset));
    ArgInfo.setInReg(true);
    return ArgInfo;
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  // All integral types are promoted to the GPR width.
  if (Ty->isIntegralOrEnumerationType())
    return ABIArgInfo::getExtend();

  return ABIArgInfo::getDirect(
      nullptr, 0, IsO32 ? nullptr : getPaddingType(OrigOffset, CurrOffset));
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

  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  // O32 doesn't treat zero-sized structs differently from other structs.
  // However, N32/N64 ignores zero sized return values.
  if (!IsO32 && Size == 0)
    return ABIArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy) || RetTy->isVectorType()) {
    if (Size <= 128) {
      if (RetTy->isAnyComplexType())
        return ABIArgInfo::getDirect();

      // O32 returns integer vectors in registers and N32/N64 returns all small
      // aggregates in registers.
      if (!IsO32 ||
          (RetTy->isVectorType() && !RetTy->hasFloatingRepresentation())) {
        ABIArgInfo ArgInfo =
            ABIArgInfo::getDirect(returnAggregateInRegs(RetTy, Size));
        ArgInfo.setInReg(true);
        return ArgInfo;
      }
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
  if (!getCXXABI().classifyReturnType(FI))
    RetInfo = classifyReturnType(FI.getReturnType());

  // Check if a pointer to an aggregate is passed as a hidden argument.  
  uint64_t Offset = RetInfo.isIndirect() ? MinABIStackAlignInBytes : 0;

  for (auto &I : FI.arguments())
    I.info = classifyArgumentType(I.type, Offset);
}

llvm::Value* MipsABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                    CodeGenFunction &CGF) const {
  llvm::Type *BP = CGF.Int8PtrTy;
  llvm::Type *BPP = CGF.Int8PtrPtrTy;

  // Integer arguments are promoted to 32-bit on O32 and 64-bit on N32/N64.
  // Pointers are also promoted in the same way but this only matters for N32.
  unsigned SlotSizeInBits = IsO32 ? 32 : 64;
  unsigned PtrWidth = getTarget().getPointerWidth(0);
  if ((Ty->isIntegerType() &&
          CGF.getContext().getIntWidth(Ty) < SlotSizeInBits) ||
      (Ty->isPointerType() && PtrWidth < SlotSizeInBits)) {
    Ty = CGF.getContext().getIntTypeForBitwidth(SlotSizeInBits,
                                                Ty->isSignedIntegerType());
  }
 
  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  int64_t TypeAlign =
      std::min(getContext().getTypeAlign(Ty) / 8, StackAlignInBytes);
  llvm::Type *PTy = llvm::PointerType::getUnqual(CGF.ConvertType(Ty));
  llvm::Value *AddrTyped;
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
  unsigned ArgSizeInBits = CGF.getContext().getTypeSize(Ty);
  uint64_t Offset = llvm::RoundUpToAlignment(ArgSizeInBits / 8, TypeAlign);
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

  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const override;
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
      F->addFnAttr(llvm::Attribute::NoInline);
      const ReqdWorkGroupSizeAttr *Attr = FD->getAttr<ReqdWorkGroupSizeAttr>();
      if (Attr) {
        // Convert the reqd_work_group_size() attributes to metadata.
        llvm::LLVMContext &Context = F->getContext();
        llvm::NamedMDNode *OpenCLMetadata = 
            M.getModule().getOrInsertNamedMetadata("opencl.kernel_wg_size_info");

        SmallVector<llvm::Metadata *, 5> Operands;
        Operands.push_back(llvm::ConstantAsMetadata::get(F));

        Operands.push_back(
            llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(
                M.Int32Ty, llvm::APInt(32, Attr->getXDim()))));
        Operands.push_back(
            llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(
                M.Int32Ty, llvm::APInt(32, Attr->getYDim()))));
        Operands.push_back(
            llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(
                M.Int32Ty, llvm::APInt(32, Attr->getZDim()))));

        // Add a boolean constant operand for "required" (true) or "hint" (false)
        // for implementing the work_group_size_hint attr later. Currently 
        // always true as the hint is not yet implemented.
        Operands.push_back(
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::getTrue(Context)));
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

  void computeInfo(CGFunctionInfo &FI) const override;

  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class HexagonTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  HexagonTargetCodeGenInfo(CodeGenTypes &CGT)
    :TargetCodeGenInfo(new HexagonABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    return 29;
  }
};

}

void HexagonABIInfo::computeInfo(CGFunctionInfo &FI) const {
  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
  for (auto &I : FI.arguments())
    I.info = classifyArgumentType(I.type);
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

  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
    return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);

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

//===----------------------------------------------------------------------===//
// AMDGPU ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AMDGPUTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  AMDGPUTargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new DefaultABIInfo(CGT)) {}
  void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const override;
};

}

void AMDGPUTargetCodeGenInfo::SetTargetAttributes(
  const Decl *D,
  llvm::GlobalValue *GV,
  CodeGen::CodeGenModule &M) const {
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD)
    return;

  if (const auto Attr = FD->getAttr<AMDGPUNumVGPRAttr>()) {
    llvm::Function *F = cast<llvm::Function>(GV);
    uint32_t NumVGPR = Attr->getNumVGPR();
    if (NumVGPR != 0)
      F->addFnAttr("amdgpu_num_vgpr", llvm::utostr(NumVGPR));
  }

  if (const auto Attr = FD->getAttr<AMDGPUNumSGPRAttr>()) {
    llvm::Function *F = cast<llvm::Function>(GV);
    unsigned NumSGPR = Attr->getNumSGPR();
    if (NumSGPR != 0)
      F->addFnAttr("amdgpu_num_sgpr", llvm::utostr(NumSGPR));
  }
}


//===----------------------------------------------------------------------===//
// SPARC v9 ABI Implementation.
// Based on the SPARC Compliance Definition version 2.4.1.
//
// Function arguments a mapped to a nominal "parameter array" and promoted to
// registers depending on their type. Each argument occupies 8 or 16 bytes in
// the array, structs larger than 16 bytes are passed indirectly.
//
// One case requires special care:
//
//   struct mixed {
//     int i;
//     float f;
//   };
//
// When a struct mixed is passed by value, it only occupies 8 bytes in the
// parameter array, but the int is passed in an integer register, and the float
// is passed in a floating point register. This is represented as two arguments
// with the LLVM IR inreg attribute:
//
//   declare void f(i32 inreg %i, float inreg %f)
//
// The code generator will only allocate 4 bytes from the parameter array for
// the inreg arguments. All other arguments are allocated a multiple of 8
// bytes.
//
namespace {
class SparcV9ABIInfo : public ABIInfo {
public:
  SparcV9ABIInfo(CodeGenTypes &CGT) : ABIInfo(CGT) {}

private:
  ABIArgInfo classifyType(QualType RetTy, unsigned SizeLimit) const;
  void computeInfo(CGFunctionInfo &FI) const override;
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;

  // Coercion type builder for structs passed in registers. The coercion type
  // serves two purposes:
  //
  // 1. Pad structs to a multiple of 64 bits, so they are passed 'left-aligned'
  //    in registers.
  // 2. Expose aligned floating point elements as first-level elements, so the
  //    code generator knows to pass them in floating point registers.
  //
  // We also compute the InReg flag which indicates that the struct contains
  // aligned 32-bit floats.
  //
  struct CoerceBuilder {
    llvm::LLVMContext &Context;
    const llvm::DataLayout &DL;
    SmallVector<llvm::Type*, 8> Elems;
    uint64_t Size;
    bool InReg;

    CoerceBuilder(llvm::LLVMContext &c, const llvm::DataLayout &dl)
      : Context(c), DL(dl), Size(0), InReg(false) {}

    // Pad Elems with integers until Size is ToSize.
    void pad(uint64_t ToSize) {
      assert(ToSize >= Size && "Cannot remove elements");
      if (ToSize == Size)
        return;

      // Finish the current 64-bit word.
      uint64_t Aligned = llvm::RoundUpToAlignment(Size, 64);
      if (Aligned > Size && Aligned <= ToSize) {
        Elems.push_back(llvm::IntegerType::get(Context, Aligned - Size));
        Size = Aligned;
      }

      // Add whole 64-bit words.
      while (Size + 64 <= ToSize) {
        Elems.push_back(llvm::Type::getInt64Ty(Context));
        Size += 64;
      }

      // Final in-word padding.
      if (Size < ToSize) {
        Elems.push_back(llvm::IntegerType::get(Context, ToSize - Size));
        Size = ToSize;
      }
    }

    // Add a floating point element at Offset.
    void addFloat(uint64_t Offset, llvm::Type *Ty, unsigned Bits) {
      // Unaligned floats are treated as integers.
      if (Offset % Bits)
        return;
      // The InReg flag is only required if there are any floats < 64 bits.
      if (Bits < 64)
        InReg = true;
      pad(Offset);
      Elems.push_back(Ty);
      Size = Offset + Bits;
    }

    // Add a struct type to the coercion type, starting at Offset (in bits).
    void addStruct(uint64_t Offset, llvm::StructType *StrTy) {
      const llvm::StructLayout *Layout = DL.getStructLayout(StrTy);
      for (unsigned i = 0, e = StrTy->getNumElements(); i != e; ++i) {
        llvm::Type *ElemTy = StrTy->getElementType(i);
        uint64_t ElemOffset = Offset + Layout->getElementOffsetInBits(i);
        switch (ElemTy->getTypeID()) {
        case llvm::Type::StructTyID:
          addStruct(ElemOffset, cast<llvm::StructType>(ElemTy));
          break;
        case llvm::Type::FloatTyID:
          addFloat(ElemOffset, ElemTy, 32);
          break;
        case llvm::Type::DoubleTyID:
          addFloat(ElemOffset, ElemTy, 64);
          break;
        case llvm::Type::FP128TyID:
          addFloat(ElemOffset, ElemTy, 128);
          break;
        case llvm::Type::PointerTyID:
          if (ElemOffset % 64 == 0) {
            pad(ElemOffset);
            Elems.push_back(ElemTy);
            Size += 64;
          }
          break;
        default:
          break;
        }
      }
    }

    // Check if Ty is a usable substitute for the coercion type.
    bool isUsableType(llvm::StructType *Ty) const {
      if (Ty->getNumElements() != Elems.size())
        return false;
      for (unsigned i = 0, e = Elems.size(); i != e; ++i)
        if (Elems[i] != Ty->getElementType(i))
          return false;
      return true;
    }

    // Get the coercion type as a literal struct type.
    llvm::Type *getType() const {
      if (Elems.size() == 1)
        return Elems.front();
      else
        return llvm::StructType::get(Context, Elems);
    }
  };
};
} // end anonymous namespace

ABIArgInfo
SparcV9ABIInfo::classifyType(QualType Ty, unsigned SizeLimit) const {
  if (Ty->isVoidType())
    return ABIArgInfo::getIgnore();

  uint64_t Size = getContext().getTypeSize(Ty);

  // Anything too big to fit in registers is passed with an explicit indirect
  // pointer / sret pointer.
  if (Size > SizeLimit)
    return ABIArgInfo::getIndirect(0, /*ByVal=*/false);

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  // Integer types smaller than a register are extended.
  if (Size < 64 && Ty->isIntegerType())
    return ABIArgInfo::getExtend();

  // Other non-aggregates go in registers.
  if (!isAggregateTypeForABI(Ty))
    return ABIArgInfo::getDirect();

  // If a C++ object has either a non-trivial copy constructor or a non-trivial
  // destructor, it is passed with an explicit indirect pointer / sret pointer.
  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
    return ABIArgInfo::getIndirect(0, RAA == CGCXXABI::RAA_DirectInMemory);

  // This is a small aggregate type that should be passed in registers.
  // Build a coercion type from the LLVM struct type.
  llvm::StructType *StrTy = dyn_cast<llvm::StructType>(CGT.ConvertType(Ty));
  if (!StrTy)
    return ABIArgInfo::getDirect();

  CoerceBuilder CB(getVMContext(), getDataLayout());
  CB.addStruct(0, StrTy);
  CB.pad(llvm::RoundUpToAlignment(CB.DL.getTypeSizeInBits(StrTy), 64));

  // Try to use the original type for coercion.
  llvm::Type *CoerceTy = CB.isUsableType(StrTy) ? StrTy : CB.getType();

  if (CB.InReg)
    return ABIArgInfo::getDirectInReg(CoerceTy);
  else
    return ABIArgInfo::getDirect(CoerceTy);
}

llvm::Value *SparcV9ABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF) const {
  ABIArgInfo AI = classifyType(Ty, 16 * 8);
  llvm::Type *ArgTy = CGT.ConvertType(Ty);
  if (AI.canHaveCoerceToType() && !AI.getCoerceToType())
    AI.setCoerceToType(ArgTy);

  llvm::Type *BPP = CGF.Int8PtrPtrTy;
  CGBuilderTy &Builder = CGF.Builder;
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr, BPP, "ap");
  llvm::Value *Addr = Builder.CreateLoad(VAListAddrAsBPP, "ap.cur");
  llvm::Type *ArgPtrTy = llvm::PointerType::getUnqual(ArgTy);
  llvm::Value *ArgAddr;
  unsigned Stride;

  switch (AI.getKind()) {
  case ABIArgInfo::Expand:
  case ABIArgInfo::InAlloca:
    llvm_unreachable("Unsupported ABI kind for va_arg");

  case ABIArgInfo::Extend:
    Stride = 8;
    ArgAddr = Builder
      .CreateConstGEP1_32(Addr, 8 - getDataLayout().getTypeAllocSize(ArgTy),
                          "extend");
    break;

  case ABIArgInfo::Direct:
    Stride = getDataLayout().getTypeAllocSize(AI.getCoerceToType());
    ArgAddr = Addr;
    break;

  case ABIArgInfo::Indirect:
    Stride = 8;
    ArgAddr = Builder.CreateBitCast(Addr,
                                    llvm::PointerType::getUnqual(ArgPtrTy),
                                    "indirect");
    ArgAddr = Builder.CreateLoad(ArgAddr, "indirect.arg");
    break;

  case ABIArgInfo::Ignore:
    return llvm::UndefValue::get(ArgPtrTy);
  }

  // Update VAList.
  Addr = Builder.CreateConstGEP1_32(Addr, Stride, "ap.next");
  Builder.CreateStore(Addr, VAListAddrAsBPP);

  return Builder.CreatePointerCast(ArgAddr, ArgPtrTy, "arg.addr");
}

void SparcV9ABIInfo::computeInfo(CGFunctionInfo &FI) const {
  FI.getReturnInfo() = classifyType(FI.getReturnType(), 32 * 8);
  for (auto &I : FI.arguments())
    I.info = classifyType(I.type, 16 * 8);
}

namespace {
class SparcV9TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  SparcV9TargetCodeGenInfo(CodeGenTypes &CGT)
    : TargetCodeGenInfo(new SparcV9ABIInfo(CGT)) {}

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    return 14;
  }

  bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                               llvm::Value *Address) const override;
};
} // end anonymous namespace

bool
SparcV9TargetCodeGenInfo::initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                                llvm::Value *Address) const {
  // This is calculated from the LLVM and GCC tables and verified
  // against gcc output.  AFAIK all ABIs use the same encoding.

  CodeGen::CGBuilderTy &Builder = CGF.Builder;

  llvm::IntegerType *i8 = CGF.Int8Ty;
  llvm::Value *Four8 = llvm::ConstantInt::get(i8, 4);
  llvm::Value *Eight8 = llvm::ConstantInt::get(i8, 8);

  // 0-31: the 8-byte general-purpose registers
  AssignToArrayRange(Builder, Address, Eight8, 0, 31);

  // 32-63: f0-31, the 4-byte floating-point registers
  AssignToArrayRange(Builder, Address, Four8, 32, 63);

  //   Y   = 64
  //   PSR = 65
  //   WIM = 66
  //   TBR = 67
  //   PC  = 68
  //   NPC = 69
  //   FSR = 70
  //   CSR = 71
  AssignToArrayRange(Builder, Address, Eight8, 64, 71);
   
  // 72-87: d0-15, the 8-byte floating-point registers
  AssignToArrayRange(Builder, Address, Eight8, 72, 87);

  return false;
}


//===----------------------------------------------------------------------===//
// XCore ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

/// A SmallStringEnc instance is used to build up the TypeString by passing
/// it by reference between functions that append to it.
typedef llvm::SmallString<128> SmallStringEnc;

/// TypeStringCache caches the meta encodings of Types.
///
/// The reason for caching TypeStrings is two fold:
///   1. To cache a type's encoding for later uses;
///   2. As a means to break recursive member type inclusion.
///
/// A cache Entry can have a Status of:
///   NonRecursive:   The type encoding is not recursive;
///   Recursive:      The type encoding is recursive;
///   Incomplete:     An incomplete TypeString;
///   IncompleteUsed: An incomplete TypeString that has been used in a
///                   Recursive type encoding.
///
/// A NonRecursive entry will have all of its sub-members expanded as fully
/// as possible. Whilst it may contain types which are recursive, the type
/// itself is not recursive and thus its encoding may be safely used whenever
/// the type is encountered.
///
/// A Recursive entry will have all of its sub-members expanded as fully as
/// possible. The type itself is recursive and it may contain other types which
/// are recursive. The Recursive encoding must not be used during the expansion
/// of a recursive type's recursive branch. For simplicity the code uses
/// IncompleteCount to reject all usage of Recursive encodings for member types.
///
/// An Incomplete entry is always a RecordType and only encodes its
/// identifier e.g. "s(S){}". Incomplete 'StubEnc' entries are ephemeral and
/// are placed into the cache during type expansion as a means to identify and
/// handle recursive inclusion of types as sub-members. If there is recursion
/// the entry becomes IncompleteUsed.
///
/// During the expansion of a RecordType's members:
///
///   If the cache contains a NonRecursive encoding for the member type, the
///   cached encoding is used;
///
///   If the cache contains a Recursive encoding for the member type, the
///   cached encoding is 'Swapped' out, as it may be incorrect, and...
///
///   If the member is a RecordType, an Incomplete encoding is placed into the
///   cache to break potential recursive inclusion of itself as a sub-member;
///
///   Once a member RecordType has been expanded, its temporary incomplete
///   entry is removed from the cache. If a Recursive encoding was swapped out
///   it is swapped back in;
///
///   If an incomplete entry is used to expand a sub-member, the incomplete
///   entry is marked as IncompleteUsed. The cache keeps count of how many
///   IncompleteUsed entries it currently contains in IncompleteUsedCount;
///
///   If a member's encoding is found to be a NonRecursive or Recursive viz:
///   IncompleteUsedCount==0, the member's encoding is added to the cache.
///   Else the member is part of a recursive type and thus the recursion has
///   been exited too soon for the encoding to be correct for the member.
///
class TypeStringCache {
  enum Status {NonRecursive, Recursive, Incomplete, IncompleteUsed};
  struct Entry {
    std::string Str;     // The encoded TypeString for the type.
    enum Status State;   // Information about the encoding in 'Str'.
    std::string Swapped; // A temporary place holder for a Recursive encoding
                         // during the expansion of RecordType's members.
  };
  std::map<const IdentifierInfo *, struct Entry> Map;
  unsigned IncompleteCount;     // Number of Incomplete entries in the Map.
  unsigned IncompleteUsedCount; // Number of IncompleteUsed entries in the Map.
public:
  TypeStringCache() : IncompleteCount(0), IncompleteUsedCount(0) {};
  void addIncomplete(const IdentifierInfo *ID, std::string StubEnc);
  bool removeIncomplete(const IdentifierInfo *ID);
  void addIfComplete(const IdentifierInfo *ID, StringRef Str,
                     bool IsRecursive);
  StringRef lookupStr(const IdentifierInfo *ID);
};

/// TypeString encodings for enum & union fields must be order.
/// FieldEncoding is a helper for this ordering process.
class FieldEncoding {
  bool HasName;
  std::string Enc;
public:
  FieldEncoding(bool b, SmallStringEnc &e) : HasName(b), Enc(e.c_str()) {};
  StringRef str() {return Enc.c_str();};
  bool operator<(const FieldEncoding &rhs) const {
    if (HasName != rhs.HasName) return HasName;
    return Enc < rhs.Enc;
  }
};

class XCoreABIInfo : public DefaultABIInfo {
public:
  XCoreABIInfo(CodeGen::CodeGenTypes &CGT) : DefaultABIInfo(CGT) {}
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                         CodeGenFunction &CGF) const override;
};

class XCoreTargetCodeGenInfo : public TargetCodeGenInfo {
  mutable TypeStringCache TSC;
public:
  XCoreTargetCodeGenInfo(CodeGenTypes &CGT)
    :TargetCodeGenInfo(new XCoreABIInfo(CGT)) {}
  void emitTargetMD(const Decl *D, llvm::GlobalValue *GV,
                    CodeGen::CodeGenModule &M) const override;
};

} // End anonymous namespace.

llvm::Value *XCoreABIInfo::EmitVAArg(llvm::Value *VAListAddr, QualType Ty,
                                     CodeGenFunction &CGF) const {
  CGBuilderTy &Builder = CGF.Builder;

  // Get the VAList.
  llvm::Value *VAListAddrAsBPP = Builder.CreateBitCast(VAListAddr,
                                                       CGF.Int8PtrPtrTy);
  llvm::Value *AP = Builder.CreateLoad(VAListAddrAsBPP);

  // Handle the argument.
  ABIArgInfo AI = classifyArgumentType(Ty);
  llvm::Type *ArgTy = CGT.ConvertType(Ty);
  if (AI.canHaveCoerceToType() && !AI.getCoerceToType())
    AI.setCoerceToType(ArgTy);
  llvm::Type *ArgPtrTy = llvm::PointerType::getUnqual(ArgTy);
  llvm::Value *Val;
  uint64_t ArgSize = 0;
  switch (AI.getKind()) {
  case ABIArgInfo::Expand:
  case ABIArgInfo::InAlloca:
    llvm_unreachable("Unsupported ABI kind for va_arg");
  case ABIArgInfo::Ignore:
    Val = llvm::UndefValue::get(ArgPtrTy);
    ArgSize = 0;
    break;
  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    Val = Builder.CreatePointerCast(AP, ArgPtrTy);
    ArgSize = getDataLayout().getTypeAllocSize(AI.getCoerceToType());
    if (ArgSize < 4)
      ArgSize = 4;
    break;
  case ABIArgInfo::Indirect:
    llvm::Value *ArgAddr;
    ArgAddr = Builder.CreateBitCast(AP, llvm::PointerType::getUnqual(ArgPtrTy));
    ArgAddr = Builder.CreateLoad(ArgAddr);
    Val = Builder.CreatePointerCast(ArgAddr, ArgPtrTy);
    ArgSize = 4;
    break;
  }

  // Increment the VAList.
  if (ArgSize) {
    llvm::Value *APN = Builder.CreateConstGEP1_32(AP, ArgSize);
    Builder.CreateStore(APN, VAListAddrAsBPP);
  }
  return Val;
}

/// During the expansion of a RecordType, an incomplete TypeString is placed
/// into the cache as a means to identify and break recursion.
/// If there is a Recursive encoding in the cache, it is swapped out and will
/// be reinserted by removeIncomplete().
/// All other types of encoding should have been used rather than arriving here.
void TypeStringCache::addIncomplete(const IdentifierInfo *ID,
                                    std::string StubEnc) {
  if (!ID)
    return;
  Entry &E = Map[ID];
  assert( (E.Str.empty() || E.State == Recursive) &&
         "Incorrectly use of addIncomplete");
  assert(!StubEnc.empty() && "Passing an empty string to addIncomplete()");
  E.Swapped.swap(E.Str); // swap out the Recursive
  E.Str.swap(StubEnc);
  E.State = Incomplete;
  ++IncompleteCount;
}

/// Once the RecordType has been expanded, the temporary incomplete TypeString
/// must be removed from the cache.
/// If a Recursive was swapped out by addIncomplete(), it will be replaced.
/// Returns true if the RecordType was defined recursively.
bool TypeStringCache::removeIncomplete(const IdentifierInfo *ID) {
  if (!ID)
    return false;
  auto I = Map.find(ID);
  assert(I != Map.end() && "Entry not present");
  Entry &E = I->second;
  assert( (E.State == Incomplete ||
           E.State == IncompleteUsed) &&
         "Entry must be an incomplete type");
  bool IsRecursive = false;
  if (E.State == IncompleteUsed) {
    // We made use of our Incomplete encoding, thus we are recursive.
    IsRecursive = true;
    --IncompleteUsedCount;
  }
  if (E.Swapped.empty())
    Map.erase(I);
  else {
    // Swap the Recursive back.
    E.Swapped.swap(E.Str);
    E.Swapped.clear();
    E.State = Recursive;
  }
  --IncompleteCount;
  return IsRecursive;
}

/// Add the encoded TypeString to the cache only if it is NonRecursive or
/// Recursive (viz: all sub-members were expanded as fully as possible).
void TypeStringCache::addIfComplete(const IdentifierInfo *ID, StringRef Str,
                                    bool IsRecursive) {
  if (!ID || IncompleteUsedCount)
    return; // No key or it is is an incomplete sub-type so don't add.
  Entry &E = Map[ID];
  if (IsRecursive && !E.Str.empty()) {
    assert(E.State==Recursive && E.Str.size() == Str.size() &&
           "This is not the same Recursive entry");
    // The parent container was not recursive after all, so we could have used
    // this Recursive sub-member entry after all, but we assumed the worse when
    // we started viz: IncompleteCount!=0.
    return;
  }
  assert(E.Str.empty() && "Entry already present");
  E.Str = Str.str();
  E.State = IsRecursive? Recursive : NonRecursive;
}

/// Return a cached TypeString encoding for the ID. If there isn't one, or we
/// are recursively expanding a type (IncompleteCount != 0) and the cached
/// encoding is Recursive, return an empty StringRef.
StringRef TypeStringCache::lookupStr(const IdentifierInfo *ID) {
  if (!ID)
    return StringRef();   // We have no key.
  auto I = Map.find(ID);
  if (I == Map.end())
    return StringRef();   // We have no encoding.
  Entry &E = I->second;
  if (E.State == Recursive && IncompleteCount)
    return StringRef();   // We don't use Recursive encodings for member types.

  if (E.State == Incomplete) {
    // The incomplete type is being used to break out of recursion.
    E.State = IncompleteUsed;
    ++IncompleteUsedCount;
  }
  return E.Str.c_str();
}

/// The XCore ABI includes a type information section that communicates symbol
/// type information to the linker. The linker uses this information to verify
/// safety/correctness of things such as array bound and pointers et al.
/// The ABI only requires C (and XC) language modules to emit TypeStrings.
/// This type information (TypeString) is emitted into meta data for all global
/// symbols: definitions, declarations, functions & variables.
///
/// The TypeString carries type, qualifier, name, size & value details.
/// Please see 'Tools Development Guide' section 2.16.2 for format details:
/// <https://www.xmos.com/download/public/Tools-Development-Guide%28X9114A%29.pdf>
/// The output is tested by test/CodeGen/xcore-stringtype.c.
///
static bool getTypeString(SmallStringEnc &Enc, const Decl *D,
                          CodeGen::CodeGenModule &CGM, TypeStringCache &TSC);

/// XCore uses emitTargetMD to emit TypeString metadata for global symbols.
void XCoreTargetCodeGenInfo::emitTargetMD(const Decl *D, llvm::GlobalValue *GV,
                                          CodeGen::CodeGenModule &CGM) const {
  SmallStringEnc Enc;
  if (getTypeString(Enc, D, CGM, TSC)) {
    llvm::LLVMContext &Ctx = CGM.getModule().getContext();
    llvm::SmallVector<llvm::Metadata *, 2> MDVals;
    MDVals.push_back(llvm::ConstantAsMetadata::get(GV));
    MDVals.push_back(llvm::MDString::get(Ctx, Enc.str()));
    llvm::NamedMDNode *MD =
      CGM.getModule().getOrInsertNamedMetadata("xcore.typestrings");
    MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
  }
}

static bool appendType(SmallStringEnc &Enc, QualType QType,
                       const CodeGen::CodeGenModule &CGM,
                       TypeStringCache &TSC);

/// Helper function for appendRecordType().
/// Builds a SmallVector containing the encoded field types in declaration order.
static bool extractFieldType(SmallVectorImpl<FieldEncoding> &FE,
                             const RecordDecl *RD,
                             const CodeGen::CodeGenModule &CGM,
                             TypeStringCache &TSC) {
  for (const auto *Field : RD->fields()) {
    SmallStringEnc Enc;
    Enc += "m(";
    Enc += Field->getName();
    Enc += "){";
    if (Field->isBitField()) {
      Enc += "b(";
      llvm::raw_svector_ostream OS(Enc);
      OS.resync();
      OS << Field->getBitWidthValue(CGM.getContext());
      OS.flush();
      Enc += ':';
    }
    if (!appendType(Enc, Field->getType(), CGM, TSC))
      return false;
    if (Field->isBitField())
      Enc += ')';
    Enc += '}';
    FE.push_back(FieldEncoding(!Field->getName().empty(), Enc));
  }
  return true;
}

/// Appends structure and union types to Enc and adds encoding to cache.
/// Recursively calls appendType (via extractFieldType) for each field.
/// Union types have their fields ordered according to the ABI.
static bool appendRecordType(SmallStringEnc &Enc, const RecordType *RT,
                             const CodeGen::CodeGenModule &CGM,
                             TypeStringCache &TSC, const IdentifierInfo *ID) {
  // Append the cached TypeString if we have one.
  StringRef TypeString = TSC.lookupStr(ID);
  if (!TypeString.empty()) {
    Enc += TypeString;
    return true;
  }

  // Start to emit an incomplete TypeString.
  size_t Start = Enc.size();
  Enc += (RT->isUnionType()? 'u' : 's');
  Enc += '(';
  if (ID)
    Enc += ID->getName();
  Enc += "){";

  // We collect all encoded fields and order as necessary.
  bool IsRecursive = false;
  const RecordDecl *RD = RT->getDecl()->getDefinition();
  if (RD && !RD->field_empty()) {
    // An incomplete TypeString stub is placed in the cache for this RecordType
    // so that recursive calls to this RecordType will use it whilst building a
    // complete TypeString for this RecordType.
    SmallVector<FieldEncoding, 16> FE;
    std::string StubEnc(Enc.substr(Start).str());
    StubEnc += '}';  // StubEnc now holds a valid incomplete TypeString.
    TSC.addIncomplete(ID, std::move(StubEnc));
    if (!extractFieldType(FE, RD, CGM, TSC)) {
      (void) TSC.removeIncomplete(ID);
      return false;
    }
    IsRecursive = TSC.removeIncomplete(ID);
    // The ABI requires unions to be sorted but not structures.
    // See FieldEncoding::operator< for sort algorithm.
    if (RT->isUnionType())
      std::sort(FE.begin(), FE.end());
    // We can now complete the TypeString.
    unsigned E = FE.size();
    for (unsigned I = 0; I != E; ++I) {
      if (I)
        Enc += ',';
      Enc += FE[I].str();
    }
  }
  Enc += '}';
  TSC.addIfComplete(ID, Enc.substr(Start), IsRecursive);
  return true;
}

/// Appends enum types to Enc and adds the encoding to the cache.
static bool appendEnumType(SmallStringEnc &Enc, const EnumType *ET,
                           TypeStringCache &TSC,
                           const IdentifierInfo *ID) {
  // Append the cached TypeString if we have one.
  StringRef TypeString = TSC.lookupStr(ID);
  if (!TypeString.empty()) {
    Enc += TypeString;
    return true;
  }

  size_t Start = Enc.size();
  Enc += "e(";
  if (ID)
    Enc += ID->getName();
  Enc += "){";

  // We collect all encoded enumerations and order them alphanumerically.
  if (const EnumDecl *ED = ET->getDecl()->getDefinition()) {
    SmallVector<FieldEncoding, 16> FE;
    for (auto I = ED->enumerator_begin(), E = ED->enumerator_end(); I != E;
         ++I) {
      SmallStringEnc EnumEnc;
      EnumEnc += "m(";
      EnumEnc += I->getName();
      EnumEnc += "){";
      I->getInitVal().toString(EnumEnc);
      EnumEnc += '}';
      FE.push_back(FieldEncoding(!I->getName().empty(), EnumEnc));
    }
    std::sort(FE.begin(), FE.end());
    unsigned E = FE.size();
    for (unsigned I = 0; I != E; ++I) {
      if (I)
        Enc += ',';
      Enc += FE[I].str();
    }
  }
  Enc += '}';
  TSC.addIfComplete(ID, Enc.substr(Start), false);
  return true;
}

/// Appends type's qualifier to Enc.
/// This is done prior to appending the type's encoding.
static void appendQualifier(SmallStringEnc &Enc, QualType QT) {
  // Qualifiers are emitted in alphabetical order.
  static const char *Table[] = {"","c:","r:","cr:","v:","cv:","rv:","crv:"};
  int Lookup = 0;
  if (QT.isConstQualified())
    Lookup += 1<<0;
  if (QT.isRestrictQualified())
    Lookup += 1<<1;
  if (QT.isVolatileQualified())
    Lookup += 1<<2;
  Enc += Table[Lookup];
}

/// Appends built-in types to Enc.
static bool appendBuiltinType(SmallStringEnc &Enc, const BuiltinType *BT) {
  const char *EncType;
  switch (BT->getKind()) {
    case BuiltinType::Void:
      EncType = "0";
      break;
    case BuiltinType::Bool:
      EncType = "b";
      break;
    case BuiltinType::Char_U:
      EncType = "uc";
      break;
    case BuiltinType::UChar:
      EncType = "uc";
      break;
    case BuiltinType::SChar:
      EncType = "sc";
      break;
    case BuiltinType::UShort:
      EncType = "us";
      break;
    case BuiltinType::Short:
      EncType = "ss";
      break;
    case BuiltinType::UInt:
      EncType = "ui";
      break;
    case BuiltinType::Int:
      EncType = "si";
      break;
    case BuiltinType::ULong:
      EncType = "ul";
      break;
    case BuiltinType::Long:
      EncType = "sl";
      break;
    case BuiltinType::ULongLong:
      EncType = "ull";
      break;
    case BuiltinType::LongLong:
      EncType = "sll";
      break;
    case BuiltinType::Float:
      EncType = "ft";
      break;
    case BuiltinType::Double:
      EncType = "d";
      break;
    case BuiltinType::LongDouble:
      EncType = "ld";
      break;
    default:
      return false;
  }
  Enc += EncType;
  return true;
}

/// Appends a pointer encoding to Enc before calling appendType for the pointee.
static bool appendPointerType(SmallStringEnc &Enc, const PointerType *PT,
                              const CodeGen::CodeGenModule &CGM,
                              TypeStringCache &TSC) {
  Enc += "p(";
  if (!appendType(Enc, PT->getPointeeType(), CGM, TSC))
    return false;
  Enc += ')';
  return true;
}

/// Appends array encoding to Enc before calling appendType for the element.
static bool appendArrayType(SmallStringEnc &Enc, QualType QT,
                            const ArrayType *AT,
                            const CodeGen::CodeGenModule &CGM,
                            TypeStringCache &TSC, StringRef NoSizeEnc) {
  if (AT->getSizeModifier() != ArrayType::Normal)
    return false;
  Enc += "a(";
  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT))
    CAT->getSize().toStringUnsigned(Enc);
  else
    Enc += NoSizeEnc; // Global arrays use "*", otherwise it is "".
  Enc += ':';
  // The Qualifiers should be attached to the type rather than the array.
  appendQualifier(Enc, QT);
  if (!appendType(Enc, AT->getElementType(), CGM, TSC))
    return false;
  Enc += ')';
  return true;
}

/// Appends a function encoding to Enc, calling appendType for the return type
/// and the arguments.
static bool appendFunctionType(SmallStringEnc &Enc, const FunctionType *FT,
                             const CodeGen::CodeGenModule &CGM,
                             TypeStringCache &TSC) {
  Enc += "f{";
  if (!appendType(Enc, FT->getReturnType(), CGM, TSC))
    return false;
  Enc += "}(";
  if (const FunctionProtoType *FPT = FT->getAs<FunctionProtoType>()) {
    // N.B. we are only interested in the adjusted param types.
    auto I = FPT->param_type_begin();
    auto E = FPT->param_type_end();
    if (I != E) {
      do {
        if (!appendType(Enc, *I, CGM, TSC))
          return false;
        ++I;
        if (I != E)
          Enc += ',';
      } while (I != E);
      if (FPT->isVariadic())
        Enc += ",va";
    } else {
      if (FPT->isVariadic())
        Enc += "va";
      else
        Enc += '0';
    }
  }
  Enc += ')';
  return true;
}

/// Handles the type's qualifier before dispatching a call to handle specific
/// type encodings.
static bool appendType(SmallStringEnc &Enc, QualType QType,
                       const CodeGen::CodeGenModule &CGM,
                       TypeStringCache &TSC) {

  QualType QT = QType.getCanonicalType();

  if (const ArrayType *AT = QT->getAsArrayTypeUnsafe())
    // The Qualifiers should be attached to the type rather than the array.
    // Thus we don't call appendQualifier() here.
    return appendArrayType(Enc, QT, AT, CGM, TSC, "");

  appendQualifier(Enc, QT);

  if (const BuiltinType *BT = QT->getAs<BuiltinType>())
    return appendBuiltinType(Enc, BT);

  if (const PointerType *PT = QT->getAs<PointerType>())
    return appendPointerType(Enc, PT, CGM, TSC);

  if (const EnumType *ET = QT->getAs<EnumType>())
    return appendEnumType(Enc, ET, TSC, QT.getBaseTypeIdentifier());

  if (const RecordType *RT = QT->getAsStructureType())
    return appendRecordType(Enc, RT, CGM, TSC, QT.getBaseTypeIdentifier());

  if (const RecordType *RT = QT->getAsUnionType())
    return appendRecordType(Enc, RT, CGM, TSC, QT.getBaseTypeIdentifier());

  if (const FunctionType *FT = QT->getAs<FunctionType>())
    return appendFunctionType(Enc, FT, CGM, TSC);

  return false;
}

static bool getTypeString(SmallStringEnc &Enc, const Decl *D,
                          CodeGen::CodeGenModule &CGM, TypeStringCache &TSC) {
  if (!D)
    return false;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->getLanguageLinkage() != CLanguageLinkage)
      return false;
    return appendType(Enc, FD->getType(), CGM, TSC);
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->getLanguageLinkage() != CLanguageLinkage)
      return false;
    QualType QT = VD->getType().getCanonicalType();
    if (const ArrayType *AT = QT->getAsArrayTypeUnsafe()) {
      // Global ArrayTypes are given a size of '*' if the size is unknown.
      // The Qualifiers should be attached to the type rather than the array.
      // Thus we don't call appendQualifier() here.
      return appendArrayType(Enc, QT, AT, CGM, TSC, "*");
    }
    return appendType(Enc, QT, CGM, TSC);
  }
  return false;
}


//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

const llvm::Triple &CodeGenModule::getTriple() const {
  return getTarget().getTriple();
}

bool CodeGenModule::supportsCOMDAT() const {
  return !getTriple().isOSBinFormatMachO();
}

const TargetCodeGenInfo &CodeGenModule::getTargetCodeGenInfo() {
  if (TheTargetCodeGenInfo)
    return *TheTargetCodeGenInfo;

  const llvm::Triple &Triple = getTarget().getTriple();
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

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be: {
    AArch64ABIInfo::ABIKind Kind = AArch64ABIInfo::AAPCS;
    if (getTarget().getABI() == "darwinpcs")
      Kind = AArch64ABIInfo::DarwinPCS;

    return *(TheTargetCodeGenInfo = new AArch64TargetCodeGenInfo(Types, Kind));
  }

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    {
      ARMABIInfo::ABIKind Kind = ARMABIInfo::AAPCS;
      if (getTarget().getABI() == "apcs-gnu")
        Kind = ARMABIInfo::APCS;
      else if (CodeGenOpts.FloatABI == "hard" ||
               (CodeGenOpts.FloatABI != "soft" &&
                Triple.getEnvironment() == llvm::Triple::GNUEABIHF))
        Kind = ARMABIInfo::AAPCS_VFP;

      switch (Triple.getOS()) {
        case llvm::Triple::NaCl:
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
    if (Triple.isOSBinFormatELF()) {
      PPC64_SVR4_ABIInfo::ABIKind Kind = PPC64_SVR4_ABIInfo::ELFv1;
      if (getTarget().getABI() == "elfv2")
        Kind = PPC64_SVR4_ABIInfo::ELFv2;

      return *(TheTargetCodeGenInfo =
               new PPC64_SVR4_TargetCodeGenInfo(Types, Kind));
    } else
      return *(TheTargetCodeGenInfo = new PPC64TargetCodeGenInfo(Types));
  case llvm::Triple::ppc64le: {
    assert(Triple.isOSBinFormatELF() && "PPC64 LE non-ELF not supported!");
    PPC64_SVR4_ABIInfo::ABIKind Kind = PPC64_SVR4_ABIInfo::ELFv2;
    if (getTarget().getABI() == "elfv1")
      Kind = PPC64_SVR4_ABIInfo::ELFv1;

    return *(TheTargetCodeGenInfo =
             new PPC64_SVR4_TargetCodeGenInfo(Types, Kind));
  }

  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return *(TheTargetCodeGenInfo = new NVPTXTargetCodeGenInfo(Types));

  case llvm::Triple::msp430:
    return *(TheTargetCodeGenInfo = new MSP430TargetCodeGenInfo(Types));

  case llvm::Triple::systemz:
    return *(TheTargetCodeGenInfo = new SystemZTargetCodeGenInfo(Types));

  case llvm::Triple::tce:
    return *(TheTargetCodeGenInfo = new TCETargetCodeGenInfo(Types));

  case llvm::Triple::x86: {
    bool IsDarwinVectorABI = Triple.isOSDarwin();
    bool IsSmallStructInRegABI =
        X86_32TargetCodeGenInfo::isStructReturnInRegABI(Triple, CodeGenOpts);
    bool IsWin32FloatStructABI = Triple.isOSWindows() && !Triple.isOSCygMing();

    if (Triple.getOS() == llvm::Triple::Win32) {
      return *(TheTargetCodeGenInfo =
               new WinX86_32TargetCodeGenInfo(Types,
                                              IsDarwinVectorABI, IsSmallStructInRegABI,
                                              IsWin32FloatStructABI,
                                              CodeGenOpts.NumRegisterParameters));
    } else {
      return *(TheTargetCodeGenInfo =
               new X86_32TargetCodeGenInfo(Types,
                                           IsDarwinVectorABI, IsSmallStructInRegABI,
                                           IsWin32FloatStructABI,
                                           CodeGenOpts.NumRegisterParameters));
    }
  }

  case llvm::Triple::x86_64: {
    bool HasAVX = getTarget().getABI() == "avx";

    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      return *(TheTargetCodeGenInfo =
                   new WinX86_64TargetCodeGenInfo(Types, HasAVX));
    case llvm::Triple::NaCl:
      return *(TheTargetCodeGenInfo =
                   new NaClX86_64TargetCodeGenInfo(Types, HasAVX));
    default:
      return *(TheTargetCodeGenInfo =
                   new X86_64TargetCodeGenInfo(Types, HasAVX));
    }
  }
  case llvm::Triple::hexagon:
    return *(TheTargetCodeGenInfo = new HexagonTargetCodeGenInfo(Types));
  case llvm::Triple::r600:
    return *(TheTargetCodeGenInfo = new AMDGPUTargetCodeGenInfo(Types));
  case llvm::Triple::amdgcn:
    return *(TheTargetCodeGenInfo = new AMDGPUTargetCodeGenInfo(Types));
  case llvm::Triple::sparcv9:
    return *(TheTargetCodeGenInfo = new SparcV9TargetCodeGenInfo(Types));
  case llvm::Triple::xcore:
    return *(TheTargetCodeGenInfo = new XCoreTargetCodeGenInfo(Types));
  }
}
