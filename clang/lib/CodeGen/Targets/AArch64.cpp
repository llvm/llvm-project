//===- AArch64.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/TargetParser/AArch64TargetParser.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AArch64ABIInfo : public ABIInfo {
  AArch64ABIKind Kind;

public:
  AArch64ABIInfo(CodeGenTypes &CGT, AArch64ABIKind Kind)
      : ABIInfo(CGT), Kind(Kind) {}

  bool isSoftFloat() const { return Kind == AArch64ABIKind::AAPCSSoft; }

private:
  AArch64ABIKind getABIKind() const { return Kind; }
  bool isDarwinPCS() const { return Kind == AArch64ABIKind::DarwinPCS; }

  ABIArgInfo classifyReturnType(QualType RetTy, bool IsVariadicFn) const;
  ABIArgInfo classifyArgumentType(QualType RetTy, bool IsVariadicFn,
                                  bool IsNamedArg, unsigned CallingConvention,
                                  unsigned &NSRN, unsigned &NPRN) const;
  llvm::Type *convertFixedToScalableVectorType(const VectorType *VT) const;
  ABIArgInfo coerceIllegalVector(QualType Ty, unsigned &NSRN,
                                 unsigned &NPRN) const;
  ABIArgInfo coerceAndExpandPureScalableAggregate(
      QualType Ty, bool IsNamedArg, unsigned NVec, unsigned NPred,
      const SmallVectorImpl<llvm::Type *> &UnpaddedCoerceToSeq, unsigned &NSRN,
      unsigned &NPRN) const;
  bool isHomogeneousAggregateBaseType(QualType Ty) const override;
  bool isHomogeneousAggregateSmallEnough(const Type *Ty,
                                         uint64_t Members) const override;
  bool isZeroLengthBitfieldPermittedInHomogeneousAggregate() const override;

  bool isIllegalVectorType(QualType Ty) const;

  bool passAsAggregateType(QualType Ty) const;
  bool passAsPureScalableType(QualType Ty, unsigned &NV, unsigned &NP,
                              SmallVectorImpl<llvm::Type *> &CoerceToSeq) const;

  void flattenType(llvm::Type *Ty,
                   SmallVectorImpl<llvm::Type *> &Flattened) const;

  void computeInfo(CGFunctionInfo &FI) const override {
    if (!::classifyReturnType(getCXXABI(), FI, *this))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.isVariadic());

    unsigned ArgNo = 0;
    unsigned NSRN = 0, NPRN = 0;
    for (auto &it : FI.arguments()) {
      const bool IsNamedArg =
          !FI.isVariadic() || ArgNo < FI.getRequiredArgs().getNumRequiredArgs();
      ++ArgNo;
      it.info = classifyArgumentType(it.type, FI.isVariadic(), IsNamedArg,
                                     FI.getCallingConvention(), NSRN, NPRN);
    }
  }

  RValue EmitDarwinVAArg(Address VAListAddr, QualType Ty, CodeGenFunction &CGF,
                         AggValueSlot Slot) const;

  RValue EmitAAPCSVAArg(Address VAListAddr, QualType Ty, CodeGenFunction &CGF,
                        AArch64ABIKind Kind, AggValueSlot Slot) const;

  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override {
    llvm::Type *BaseTy = CGF.ConvertType(Ty);
    if (isa<llvm::ScalableVectorType>(BaseTy))
      llvm::report_fatal_error("Passing SVE types to variadic functions is "
                               "currently not supported");

    return Kind == AArch64ABIKind::Win64
               ? EmitMSVAArg(CGF, VAListAddr, Ty, Slot)
           : isDarwinPCS() ? EmitDarwinVAArg(VAListAddr, Ty, CGF, Slot)
                           : EmitAAPCSVAArg(VAListAddr, Ty, CGF, Kind, Slot);
  }

  RValue EmitMSVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                     AggValueSlot Slot) const override;

  bool allowBFloatArgsAndRet() const override {
    return getTarget().hasBFloat16Type();
  }

  using ABIInfo::appendAttributeMangling;
  void appendAttributeMangling(TargetClonesAttr *Attr, unsigned Index,
                               raw_ostream &Out) const override;
  void appendAttributeMangling(StringRef AttrStr,
                               raw_ostream &Out) const override;
};

class AArch64SwiftABIInfo : public SwiftABIInfo {
public:
  explicit AArch64SwiftABIInfo(CodeGenTypes &CGT)
      : SwiftABIInfo(CGT, /*SwiftErrorInRegister=*/true) {}

  bool isLegalVectorType(CharUnits VectorSize, llvm::Type *EltTy,
                         unsigned NumElts) const override;
};

class AArch64TargetCodeGenInfo : public TargetCodeGenInfo {
public:
  AArch64TargetCodeGenInfo(CodeGenTypes &CGT, AArch64ABIKind Kind)
      : TargetCodeGenInfo(std::make_unique<AArch64ABIInfo>(CGT, Kind)) {
    SwiftInfo = std::make_unique<AArch64SwiftABIInfo>(CGT);
  }

  StringRef getARCRetainAutoreleasedReturnValueMarker() const override {
    return "mov\tfp, fp\t\t// marker for objc_retainAutoreleaseReturnValue";
  }

  int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const override {
    return 31;
  }

  bool doesReturnSlotInterfereWithArgs() const override { return false; }

  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override {
    const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D);
    if (!FD)
      return;

    TargetInfo::BranchProtectionInfo BPI(CGM.getLangOpts());

    if (const auto *TA = FD->getAttr<TargetAttr>()) {
      ParsedTargetAttr Attr =
          CGM.getTarget().parseTargetAttr(TA->getFeaturesStr());
      if (!Attr.BranchProtection.empty()) {
        StringRef Error;
        (void)CGM.getTarget().validateBranchProtection(Attr.BranchProtection,
                                                       Attr.CPU, BPI, Error);
        assert(Error.empty());
      }
    }
    auto *Fn = cast<llvm::Function>(GV);
    setBranchProtectionFnAttributes(BPI, *Fn);
  }

  bool isScalarizableAsmOperand(CodeGen::CodeGenFunction &CGF,
                                llvm::Type *Ty) const override {
    if (CGF.getTarget().hasFeature("ls64")) {
      auto *ST = dyn_cast<llvm::StructType>(Ty);
      if (ST && ST->getNumElements() == 1) {
        auto *AT = dyn_cast<llvm::ArrayType>(ST->getElementType(0));
        if (AT && AT->getNumElements() == 8 &&
            AT->getElementType()->isIntegerTy(64))
          return true;
      }
    }
    return TargetCodeGenInfo::isScalarizableAsmOperand(CGF, Ty);
  }

  void checkFunctionABI(CodeGenModule &CGM,
                        const FunctionDecl *Decl) const override;

  void checkFunctionCallABI(CodeGenModule &CGM, SourceLocation CallLoc,
                            const FunctionDecl *Caller,
                            const FunctionDecl *Callee, const CallArgList &Args,
                            QualType ReturnType) const override;

  bool wouldInliningViolateFunctionCallABI(
      const FunctionDecl *Caller, const FunctionDecl *Callee) const override;

private:
  // Diagnose calls between functions with incompatible Streaming SVE
  // attributes.
  void checkFunctionCallABIStreaming(CodeGenModule &CGM, SourceLocation CallLoc,
                                     const FunctionDecl *Caller,
                                     const FunctionDecl *Callee) const;
  // Diagnose calls which must pass arguments in floating-point registers when
  // the selected target does not have floating-point registers.
  void checkFunctionCallABISoftFloat(CodeGenModule &CGM, SourceLocation CallLoc,
                                     const FunctionDecl *Caller,
                                     const FunctionDecl *Callee,
                                     const CallArgList &Args,
                                     QualType ReturnType) const;
};

class WindowsAArch64TargetCodeGenInfo : public AArch64TargetCodeGenInfo {
public:
  WindowsAArch64TargetCodeGenInfo(CodeGenTypes &CGT, AArch64ABIKind K)
      : AArch64TargetCodeGenInfo(CGT, K) {}

  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override;

  void getDependentLibraryOption(llvm::StringRef Lib,
                                 llvm::SmallString<24> &Opt) const override {
    Opt = "/DEFAULTLIB:" + qualifyWindowsLibrary(Lib);
  }

  void getDetectMismatchOption(llvm::StringRef Name, llvm::StringRef Value,
                               llvm::SmallString<32> &Opt) const override {
    Opt = "/FAILIFMISMATCH:\"" + Name.str() + "=" + Value.str() + "\"";
  }
};

void WindowsAArch64TargetCodeGenInfo::setTargetAttributes(
    const Decl *D, llvm::GlobalValue *GV, CodeGen::CodeGenModule &CGM) const {
  AArch64TargetCodeGenInfo::setTargetAttributes(D, GV, CGM);
  if (GV->isDeclaration())
    return;
  addStackProbeTargetAttributes(D, GV, CGM);
}
}

llvm::Type *
AArch64ABIInfo::convertFixedToScalableVectorType(const VectorType *VT) const {
  assert(VT->getElementType()->isBuiltinType() && "expected builtin type!");

  if (VT->getVectorKind() == VectorKind::SveFixedLengthPredicate) {
    assert(VT->getElementType()->castAs<BuiltinType>()->getKind() ==
               BuiltinType::UChar &&
           "unexpected builtin type for SVE predicate!");
    return llvm::ScalableVectorType::get(llvm::Type::getInt1Ty(getVMContext()),
                                         16);
  }

  if (VT->getVectorKind() == VectorKind::SveFixedLengthData) {
    const auto *BT = VT->getElementType()->castAs<BuiltinType>();
    switch (BT->getKind()) {
    default:
      llvm_unreachable("unexpected builtin type for SVE vector!");

    case BuiltinType::SChar:
    case BuiltinType::UChar:
      return llvm::ScalableVectorType::get(
          llvm::Type::getInt8Ty(getVMContext()), 16);

    case BuiltinType::Short:
    case BuiltinType::UShort:
      return llvm::ScalableVectorType::get(
          llvm::Type::getInt16Ty(getVMContext()), 8);

    case BuiltinType::Int:
    case BuiltinType::UInt:
      return llvm::ScalableVectorType::get(
          llvm::Type::getInt32Ty(getVMContext()), 4);

    case BuiltinType::Long:
    case BuiltinType::ULong:
      return llvm::ScalableVectorType::get(
          llvm::Type::getInt64Ty(getVMContext()), 2);

    case BuiltinType::Half:
      return llvm::ScalableVectorType::get(
          llvm::Type::getHalfTy(getVMContext()), 8);

    case BuiltinType::Float:
      return llvm::ScalableVectorType::get(
          llvm::Type::getFloatTy(getVMContext()), 4);

    case BuiltinType::Double:
      return llvm::ScalableVectorType::get(
          llvm::Type::getDoubleTy(getVMContext()), 2);

    case BuiltinType::BFloat16:
      return llvm::ScalableVectorType::get(
          llvm::Type::getBFloatTy(getVMContext()), 8);
    }
  }

  llvm_unreachable("expected fixed-length SVE vector");
}

ABIArgInfo AArch64ABIInfo::coerceIllegalVector(QualType Ty, unsigned &NSRN,
                                               unsigned &NPRN) const {
  assert(Ty->isVectorType() && "expected vector type!");

  const auto *VT = Ty->castAs<VectorType>();
  if (VT->getVectorKind() == VectorKind::SveFixedLengthPredicate) {
    assert(VT->getElementType()->isBuiltinType() && "expected builtin type!");
    assert(VT->getElementType()->castAs<BuiltinType>()->getKind() ==
               BuiltinType::UChar &&
           "unexpected builtin type for SVE predicate!");
    NPRN = std::min(NPRN + 1, 4u);
    return ABIArgInfo::getDirect(llvm::ScalableVectorType::get(
        llvm::Type::getInt1Ty(getVMContext()), 16));
  }

  if (VT->getVectorKind() == VectorKind::SveFixedLengthData) {
    NSRN = std::min(NSRN + 1, 8u);
    return ABIArgInfo::getDirect(convertFixedToScalableVectorType(VT));
  }

  uint64_t Size = getContext().getTypeSize(Ty);
  // Android promotes <2 x i8> to i16, not i32
  if ((isAndroid() || isOHOSFamily()) && (Size <= 16)) {
    llvm::Type *ResType = llvm::Type::getInt16Ty(getVMContext());
    return ABIArgInfo::getDirect(ResType);
  }
  if (Size <= 32) {
    llvm::Type *ResType = llvm::Type::getInt32Ty(getVMContext());
    return ABIArgInfo::getDirect(ResType);
  }
  if (Size == 64) {
    NSRN = std::min(NSRN + 1, 8u);
    auto *ResType =
        llvm::FixedVectorType::get(llvm::Type::getInt32Ty(getVMContext()), 2);
    return ABIArgInfo::getDirect(ResType);
  }
  if (Size == 128) {
    NSRN = std::min(NSRN + 1, 8u);
    auto *ResType =
        llvm::FixedVectorType::get(llvm::Type::getInt32Ty(getVMContext()), 4);
    return ABIArgInfo::getDirect(ResType);
  }

  return getNaturalAlignIndirect(Ty, /*ByVal=*/false);
}

ABIArgInfo AArch64ABIInfo::coerceAndExpandPureScalableAggregate(
    QualType Ty, bool IsNamedArg, unsigned NVec, unsigned NPred,
    const SmallVectorImpl<llvm::Type *> &UnpaddedCoerceToSeq, unsigned &NSRN,
    unsigned &NPRN) const {
  if (!IsNamedArg || NSRN + NVec > 8 || NPRN + NPred > 4)
    return getNaturalAlignIndirect(Ty, /*ByVal=*/false);
  NSRN += NVec;
  NPRN += NPred;

  // Handle SVE vector tuples.
  if (Ty->isSVESizelessBuiltinType())
    return ABIArgInfo::getDirect();

  llvm::Type *UnpaddedCoerceToType =
      UnpaddedCoerceToSeq.size() == 1
          ? UnpaddedCoerceToSeq[0]
          : llvm::StructType::get(CGT.getLLVMContext(), UnpaddedCoerceToSeq,
                                  true);

  SmallVector<llvm::Type *> CoerceToSeq;
  flattenType(CGT.ConvertType(Ty), CoerceToSeq);
  auto *CoerceToType =
      llvm::StructType::get(CGT.getLLVMContext(), CoerceToSeq, false);

  return ABIArgInfo::getCoerceAndExpand(CoerceToType, UnpaddedCoerceToType);
}

ABIArgInfo AArch64ABIInfo::classifyArgumentType(QualType Ty, bool IsVariadicFn,
                                                bool IsNamedArg,
                                                unsigned CallingConvention,
                                                unsigned &NSRN,
                                                unsigned &NPRN) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  // Handle illegal vector types here.
  if (isIllegalVectorType(Ty))
    return coerceIllegalVector(Ty, NSRN, NPRN);

  if (!passAsAggregateType(Ty)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
      Ty = EnumTy->getDecl()->getIntegerType();

    if (const auto *EIT = Ty->getAs<BitIntType>())
      if (EIT->getNumBits() > 128)
        return getNaturalAlignIndirect(Ty, false);

    if (Ty->isVectorType())
      NSRN = std::min(NSRN + 1, 8u);
    else if (const auto *BT = Ty->getAs<BuiltinType>()) {
      if (BT->isFloatingPoint())
        NSRN = std::min(NSRN + 1, 8u);
      else {
        switch (BT->getKind()) {
        case BuiltinType::MFloat8x8:
        case BuiltinType::MFloat8x16:
          NSRN = std::min(NSRN + 1, 8u);
          break;
        case BuiltinType::SveBool:
        case BuiltinType::SveCount:
          NPRN = std::min(NPRN + 1, 4u);
          break;
        case BuiltinType::SveBoolx2:
          NPRN = std::min(NPRN + 2, 4u);
          break;
        case BuiltinType::SveBoolx4:
          NPRN = std::min(NPRN + 4, 4u);
          break;
        default:
          if (BT->isSVESizelessBuiltinType())
            NSRN = std::min(
                NSRN + getContext().getBuiltinVectorTypeInfo(BT).NumVectors,
                8u);
        }
      }
    }

    return (isPromotableIntegerTypeForABI(Ty) && isDarwinPCS()
                ? ABIArgInfo::getExtend(Ty, CGT.ConvertType(Ty))
                : ABIArgInfo::getDirect());
  }

  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always indirect.
  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI())) {
    return getNaturalAlignIndirect(Ty, /*ByVal=*/RAA ==
                                     CGCXXABI::RAA_DirectInMemory);
  }

  // Empty records are always ignored on Darwin, but actually passed in C++ mode
  // elsewhere for GNU compatibility.
  uint64_t Size = getContext().getTypeSize(Ty);
  bool IsEmpty = isEmptyRecord(getContext(), Ty, true);
  if (!Ty->isSVESizelessBuiltinType() && (IsEmpty || Size == 0)) {
    if (!getContext().getLangOpts().CPlusPlus || isDarwinPCS())
      return ABIArgInfo::getIgnore();

    // GNU C mode. The only argument that gets ignored is an empty one with size
    // 0.
    if (IsEmpty && Size == 0)
      return ABIArgInfo::getIgnore();
    return ABIArgInfo::getDirect(llvm::Type::getInt8Ty(getVMContext()));
  }

  // Homogeneous Floating-point Aggregates (HFAs) need to be expanded.
  const Type *Base = nullptr;
  uint64_t Members = 0;
  bool IsWin64 = Kind == AArch64ABIKind::Win64 ||
                 CallingConvention == llvm::CallingConv::Win64;
  bool IsWinVariadic = IsWin64 && IsVariadicFn;
  // In variadic functions on Windows, all composite types are treated alike,
  // no special handling of HFAs/HVAs.
  if (!IsWinVariadic && isHomogeneousAggregate(Ty, Base, Members)) {
    NSRN = std::min(NSRN + Members, uint64_t(8));
    if (Kind != AArch64ABIKind::AAPCS)
      return ABIArgInfo::getDirect(
          llvm::ArrayType::get(CGT.ConvertType(QualType(Base, 0)), Members));

    // For HFAs/HVAs, cap the argument alignment to 16, otherwise
    // set it to 8 according to the AAPCS64 document.
    unsigned Align =
        getContext().getTypeUnadjustedAlignInChars(Ty).getQuantity();
    Align = (Align >= 16) ? 16 : 8;
    return ABIArgInfo::getDirect(
        llvm::ArrayType::get(CGT.ConvertType(QualType(Base, 0)), Members), 0,
        nullptr, true, Align);
  }

  // In AAPCS named arguments of a Pure Scalable Type are passed expanded in
  // registers, or indirectly if there are not enough registers.
  if (Kind == AArch64ABIKind::AAPCS) {
    unsigned NVec = 0, NPred = 0;
    SmallVector<llvm::Type *> UnpaddedCoerceToSeq;
    if (passAsPureScalableType(Ty, NVec, NPred, UnpaddedCoerceToSeq) &&
        (NVec + NPred) > 0)
      return coerceAndExpandPureScalableAggregate(
          Ty, IsNamedArg, NVec, NPred, UnpaddedCoerceToSeq, NSRN, NPRN);
  }

  // Aggregates <= 16 bytes are passed directly in registers or on the stack.
  if (Size <= 128) {
    unsigned Alignment;
    if (Kind == AArch64ABIKind::AAPCS) {
      Alignment = getContext().getTypeUnadjustedAlign(Ty);
      Alignment = Alignment < 128 ? 64 : 128;
    } else {
      Alignment =
          std::max(getContext().getTypeAlign(Ty),
                   (unsigned)getTarget().getPointerWidth(LangAS::Default));
    }
    Size = llvm::alignTo(Size, Alignment);

    // We use a pair of i64 for 16-byte aggregate with 8-byte alignment.
    // For aggregates with 16-byte alignment, we use i128.
    llvm::Type *BaseTy = llvm::Type::getIntNTy(getVMContext(), Alignment);
    return ABIArgInfo::getDirect(
        Size == Alignment ? BaseTy
                          : llvm::ArrayType::get(BaseTy, Size / Alignment));
  }

  return getNaturalAlignIndirect(Ty, /*ByVal=*/false);
}

ABIArgInfo AArch64ABIInfo::classifyReturnType(QualType RetTy,
                                              bool IsVariadicFn) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (const auto *VT = RetTy->getAs<VectorType>()) {
    if (VT->getVectorKind() == VectorKind::SveFixedLengthData ||
        VT->getVectorKind() == VectorKind::SveFixedLengthPredicate) {
      unsigned NSRN = 0, NPRN = 0;
      return coerceIllegalVector(RetTy, NSRN, NPRN);
    }
  }

  // Large vector types should be returned via memory.
  if (RetTy->isVectorType() && getContext().getTypeSize(RetTy) > 128)
    return getNaturalAlignIndirect(RetTy);

  if (!passAsAggregateType(RetTy)) {
    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
      RetTy = EnumTy->getDecl()->getIntegerType();

    if (const auto *EIT = RetTy->getAs<BitIntType>())
      if (EIT->getNumBits() > 128)
        return getNaturalAlignIndirect(RetTy);

    return (isPromotableIntegerTypeForABI(RetTy) && isDarwinPCS()
                ? ABIArgInfo::getExtend(RetTy)
                : ABIArgInfo::getDirect());
  }

  uint64_t Size = getContext().getTypeSize(RetTy);
  if (!RetTy->isSVESizelessBuiltinType() &&
      (isEmptyRecord(getContext(), RetTy, true) || Size == 0))
    return ABIArgInfo::getIgnore();

  const Type *Base = nullptr;
  uint64_t Members = 0;
  if (isHomogeneousAggregate(RetTy, Base, Members) &&
      !(getTarget().getTriple().getArch() == llvm::Triple::aarch64_32 &&
        IsVariadicFn))
    // Homogeneous Floating-point Aggregates (HFAs) are returned directly.
    return ABIArgInfo::getDirect();

  // In AAPCS return values of a Pure Scalable type are treated as a single
  // named argument and passed expanded in registers, or indirectly if there are
  // not enough registers.
  if (Kind == AArch64ABIKind::AAPCS) {
    unsigned NSRN = 0, NPRN = 0;
    unsigned NVec = 0, NPred = 0;
    SmallVector<llvm::Type *> UnpaddedCoerceToSeq;
    if (passAsPureScalableType(RetTy, NVec, NPred, UnpaddedCoerceToSeq) &&
        (NVec + NPred) > 0)
      return coerceAndExpandPureScalableAggregate(
          RetTy, /* IsNamedArg */ true, NVec, NPred, UnpaddedCoerceToSeq, NSRN,
          NPRN);
  }

  // Aggregates <= 16 bytes are returned directly in registers or on the stack.
  if (Size <= 128) {
    if (Size <= 64 && getDataLayout().isLittleEndian()) {
      // Composite types are returned in lower bits of a 64-bit register for LE,
      // and in higher bits for BE. However, integer types are always returned
      // in lower bits for both LE and BE, and they are not rounded up to
      // 64-bits. We can skip rounding up of composite types for LE, but not for
      // BE, otherwise composite types will be indistinguishable from integer
      // types.
      return ABIArgInfo::getDirect(
          llvm::IntegerType::get(getVMContext(), Size));
    }

    unsigned Alignment = getContext().getTypeAlign(RetTy);
    Size = llvm::alignTo(Size, 64); // round up to multiple of 8 bytes

    // We use a pair of i64 for 16-byte aggregate with 8-byte alignment.
    // For aggregates with 16-byte alignment, we use i128.
    if (Alignment < 128 && Size == 128) {
      llvm::Type *BaseTy = llvm::Type::getInt64Ty(getVMContext());
      return ABIArgInfo::getDirect(llvm::ArrayType::get(BaseTy, Size / 64));
    }
    return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), Size));
  }

  return getNaturalAlignIndirect(RetTy);
}

/// isIllegalVectorType - check whether the vector type is legal for AArch64.
bool AArch64ABIInfo::isIllegalVectorType(QualType Ty) const {
  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // Check whether VT is a fixed-length SVE vector. These types are
    // represented as scalable vectors in function args/return and must be
    // coerced from fixed vectors.
    if (VT->getVectorKind() == VectorKind::SveFixedLengthData ||
        VT->getVectorKind() == VectorKind::SveFixedLengthPredicate)
      return true;

    // Check whether VT is legal.
    unsigned NumElements = VT->getNumElements();
    uint64_t Size = getContext().getTypeSize(VT);
    // NumElements should be power of 2.
    if (!llvm::isPowerOf2_32(NumElements))
      return true;

    // arm64_32 has to be compatible with the ARM logic here, which allows huge
    // vectors for some reason.
    llvm::Triple Triple = getTarget().getTriple();
    if (Triple.getArch() == llvm::Triple::aarch64_32 &&
        Triple.isOSBinFormatMachO())
      return Size <= 32;

    return Size != 64 && (Size != 128 || NumElements == 1);
  }
  return false;
}

bool AArch64SwiftABIInfo::isLegalVectorType(CharUnits VectorSize,
                                            llvm::Type *EltTy,
                                            unsigned NumElts) const {
  if (!llvm::isPowerOf2_32(NumElts))
    return false;
  if (VectorSize.getQuantity() != 8 &&
      (VectorSize.getQuantity() != 16 || NumElts == 1))
    return false;
  return true;
}

bool AArch64ABIInfo::isHomogeneousAggregateBaseType(QualType Ty) const {
  // For the soft-float ABI variant, no types are considered to be homogeneous
  // aggregates.
  if (isSoftFloat())
    return false;

  // Homogeneous aggregates for AAPCS64 must have base types of a floating
  // point type or a short-vector type. This is the same as the 32-bit ABI,
  // but with the difference that any floating-point type is allowed,
  // including __fp16.
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    if (BT->isFloatingPoint() || BT->getKind() == BuiltinType::MFloat8x16 ||
        BT->getKind() == BuiltinType::MFloat8x8)
      return true;
  } else if (const VectorType *VT = Ty->getAs<VectorType>()) {
    if (auto Kind = VT->getVectorKind();
        Kind == VectorKind::SveFixedLengthData ||
        Kind == VectorKind::SveFixedLengthPredicate)
      return false;

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

bool AArch64ABIInfo::isZeroLengthBitfieldPermittedInHomogeneousAggregate()
    const {
  // AAPCS64 says that the rule for whether something is a homogeneous
  // aggregate is applied to the output of the data layout decision. So
  // anything that doesn't affect the data layout also does not affect
  // homogeneity. In particular, zero-length bitfields don't stop a struct
  // being homogeneous.
  return true;
}

bool AArch64ABIInfo::passAsAggregateType(QualType Ty) const {
  if (Kind == AArch64ABIKind::AAPCS && Ty->isSVESizelessBuiltinType()) {
    const auto *BT = Ty->castAs<BuiltinType>();
    return !BT->isSVECount() &&
           getContext().getBuiltinVectorTypeInfo(BT).NumVectors > 1;
  }
  return isAggregateTypeForABI(Ty);
}

// Check if a type needs to be passed in registers as a Pure Scalable Type (as
// defined by AAPCS64). Return the number of data vectors and the number of
// predicate vectors in the type, into `NVec` and `NPred`, respectively. Upon
// return `CoerceToSeq` contains an expanded sequence of LLVM IR types, one
// element for each non-composite member. For practical purposes, limit the
// length of `CoerceToSeq` to about 12 (the maximum that could possibly fit
// in registers) and return false, the effect of which will be to  pass the
// argument under the rules for a large (> 128 bytes) composite.
bool AArch64ABIInfo::passAsPureScalableType(
    QualType Ty, unsigned &NVec, unsigned &NPred,
    SmallVectorImpl<llvm::Type *> &CoerceToSeq) const {
  if (const ConstantArrayType *AT = getContext().getAsConstantArrayType(Ty)) {
    uint64_t NElt = AT->getZExtSize();
    if (NElt == 0)
      return false;

    unsigned NV = 0, NP = 0;
    SmallVector<llvm::Type *> EltCoerceToSeq;
    if (!passAsPureScalableType(AT->getElementType(), NV, NP, EltCoerceToSeq))
      return false;

    if (CoerceToSeq.size() + NElt * EltCoerceToSeq.size() > 12)
      return false;

    for (uint64_t I = 0; I < NElt; ++I)
      llvm::copy(EltCoerceToSeq, std::back_inserter(CoerceToSeq));

    NVec += NElt * NV;
    NPred += NElt * NP;
    return true;
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    // If the record cannot be passed in registers, then it's not a PST.
    if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(RT, getCXXABI());
        RAA != CGCXXABI::RAA_Default)
      return false;

    // Pure scalable types are never unions and never contain unions.
    const RecordDecl *RD = RT->getDecl();
    if (RD->isUnion())
      return false;

    // If this is a C++ record, check the bases.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      for (const auto &I : CXXRD->bases()) {
        if (isEmptyRecord(getContext(), I.getType(), true))
          continue;
        if (!passAsPureScalableType(I.getType(), NVec, NPred, CoerceToSeq))
          return false;
      }
    }

    // Check members.
    for (const auto *FD : RD->fields()) {
      QualType FT = FD->getType();
      if (isEmptyField(getContext(), FD, /* AllowArrays */ true))
        continue;
      if (!passAsPureScalableType(FT, NVec, NPred, CoerceToSeq))
        return false;
    }

    return true;
  }

  if (const auto *VT = Ty->getAs<VectorType>()) {
    if (VT->getVectorKind() == VectorKind::SveFixedLengthPredicate) {
      ++NPred;
      if (CoerceToSeq.size() + 1 > 12)
        return false;
      CoerceToSeq.push_back(convertFixedToScalableVectorType(VT));
      return true;
    }

    if (VT->getVectorKind() == VectorKind::SveFixedLengthData) {
      ++NVec;
      if (CoerceToSeq.size() + 1 > 12)
        return false;
      CoerceToSeq.push_back(convertFixedToScalableVectorType(VT));
      return true;
    }

    return false;
  }

  if (!Ty->isBuiltinType())
    return false;

  bool isPredicate;
  switch (Ty->getAs<BuiltinType>()->getKind()) {
#define SVE_VECTOR_TYPE(Name, MangledName, Id, SingletonId)                    \
  case BuiltinType::Id:                                                        \
    isPredicate = false;                                                       \
    break;
#define SVE_PREDICATE_TYPE(Name, MangledName, Id, SingletonId)                 \
  case BuiltinType::Id:                                                        \
    isPredicate = true;                                                        \
    break;
#define SVE_TYPE(Name, Id, SingletonId)
#include "clang/Basic/AArch64SVEACLETypes.def"
  default:
    return false;
  }

  ASTContext::BuiltinVectorTypeInfo Info =
      getContext().getBuiltinVectorTypeInfo(cast<BuiltinType>(Ty));
  assert(Info.NumVectors > 0 && Info.NumVectors <= 4 &&
         "Expected 1, 2, 3 or 4 vectors!");
  if (isPredicate)
    NPred += Info.NumVectors;
  else
    NVec += Info.NumVectors;
  auto VTy = llvm::ScalableVectorType::get(CGT.ConvertType(Info.ElementType),
                                           Info.EC.getKnownMinValue());

  if (CoerceToSeq.size() + Info.NumVectors > 12)
    return false;
  std::fill_n(std::back_inserter(CoerceToSeq), Info.NumVectors, VTy);

  return true;
}

// Expand an LLVM IR type into a sequence with a element for each non-struct,
// non-array member of the type, with the exception of the padding types, which
// are retained.
void AArch64ABIInfo::flattenType(
    llvm::Type *Ty, SmallVectorImpl<llvm::Type *> &Flattened) const {

  if (ABIArgInfo::isPaddingForCoerceAndExpand(Ty)) {
    Flattened.push_back(Ty);
    return;
  }

  if (const auto *AT = dyn_cast<llvm::ArrayType>(Ty)) {
    uint64_t NElt = AT->getNumElements();
    if (NElt == 0)
      return;

    SmallVector<llvm::Type *> EltFlattened;
    flattenType(AT->getElementType(), EltFlattened);

    for (uint64_t I = 0; I < NElt; ++I)
      llvm::copy(EltFlattened, std::back_inserter(Flattened));
    return;
  }

  if (const auto *ST = dyn_cast<llvm::StructType>(Ty)) {
    for (auto *ET : ST->elements())
      flattenType(ET, Flattened);
    return;
  }

  Flattened.push_back(Ty);
}

RValue AArch64ABIInfo::EmitAAPCSVAArg(Address VAListAddr, QualType Ty,
                                      CodeGenFunction &CGF, AArch64ABIKind Kind,
                                      AggValueSlot Slot) const {
  // These numbers are not used for variadic arguments, hence it doesn't matter
  // they don't retain their values across multiple calls to
  // `classifyArgumentType` here.
  unsigned NSRN = 0, NPRN = 0;
  ABIArgInfo AI =
      classifyArgumentType(Ty, /*IsVariadicFn=*/true, /* IsNamedArg */ false,
                           CGF.CurFnInfo->getCallingConvention(), NSRN, NPRN);
  // Empty records are ignored for parameter passing purposes.
  if (AI.isIgnore())
    return Slot.asRValue();

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
  bool IsFPR =
      !isSoftFloat() && (BaseTy->isFloatingPointTy() || BaseTy->isVectorTy());

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

  CharUnits TySize = getContext().getTypeSizeInChars(Ty);
  CharUnits TyAlign = getContext().getTypeUnadjustedAlignInChars(Ty);

  Address reg_offs_p = Address::invalid();
  llvm::Value *reg_offs = nullptr;
  int reg_top_index;
  int RegSize = IsIndirect ? 8 : TySize.getQuantity();
  if (!IsFPR) {
    // 3 is the field number of __gr_offs
    reg_offs_p = CGF.Builder.CreateStructGEP(VAListAddr, 3, "gr_offs_p");
    reg_offs = CGF.Builder.CreateLoad(reg_offs_p, "gr_offs");
    reg_top_index = 1; // field number for __gr_top
    RegSize = llvm::alignTo(RegSize, 8);
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
  if (!IsFPR && !IsIndirect && TyAlign.getQuantity() > 8) {
    int Align = TyAlign.getQuantity();

    reg_offs = CGF.Builder.CreateAdd(
        reg_offs, llvm::ConstantInt::get(CGF.Int32Ty, Align - 1),
        "align_regoffs");
    reg_offs = CGF.Builder.CreateAnd(
        reg_offs, llvm::ConstantInt::get(CGF.Int32Ty, -Align),
        "aligned_regoffs");
  }

  // Update the gr_offs/vr_offs pointer for next call to va_arg on this va_list.
  // The fact that this is done unconditionally reflects the fact that
  // allocating an argument to the stack also uses up all the remaining
  // registers of the appropriate kind.
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

  llvm::Value *reg_top = nullptr;
  Address reg_top_p =
      CGF.Builder.CreateStructGEP(VAListAddr, reg_top_index, "reg_top_p");
  reg_top = CGF.Builder.CreateLoad(reg_top_p, "reg_top");
  Address BaseAddr(CGF.Builder.CreateInBoundsGEP(CGF.Int8Ty, reg_top, reg_offs),
                   CGF.Int8Ty, CharUnits::fromQuantity(IsFPR ? 16 : 8));
  Address RegAddr = Address::invalid();
  llvm::Type *MemTy = CGF.ConvertTypeForMem(Ty), *ElementTy = MemTy;

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
    auto BaseTyInfo = getContext().getTypeInfoInChars(QualType(Base, 0));
    llvm::Type *BaseTy = CGF.ConvertType(QualType(Base, 0));
    llvm::Type *HFATy = llvm::ArrayType::get(BaseTy, NumMembers);
    Address Tmp = CGF.CreateTempAlloca(HFATy,
                                       std::max(TyAlign, BaseTyInfo.Align));

    // On big-endian platforms, the value will be right-aligned in its slot.
    int Offset = 0;
    if (CGF.CGM.getDataLayout().isBigEndian() &&
        BaseTyInfo.Width.getQuantity() < 16)
      Offset = 16 - BaseTyInfo.Width.getQuantity();

    for (unsigned i = 0; i < NumMembers; ++i) {
      CharUnits BaseOffset = CharUnits::fromQuantity(16 * i + Offset);
      Address LoadAddr =
        CGF.Builder.CreateConstInBoundsByteGEP(BaseAddr, BaseOffset);
      LoadAddr = LoadAddr.withElementType(BaseTy);

      Address StoreAddr = CGF.Builder.CreateConstArrayGEP(Tmp, i);

      llvm::Value *Elem = CGF.Builder.CreateLoad(LoadAddr);
      CGF.Builder.CreateStore(Elem, StoreAddr);
    }

    RegAddr = Tmp.withElementType(MemTy);
  } else {
    // Otherwise the object is contiguous in memory.

    // It might be right-aligned in its slot.
    CharUnits SlotSize = BaseAddr.getAlignment();
    if (CGF.CGM.getDataLayout().isBigEndian() && !IsIndirect &&
        (IsHFA || !isAggregateTypeForABI(Ty)) &&
        TySize < SlotSize) {
      CharUnits Offset = SlotSize - TySize;
      BaseAddr = CGF.Builder.CreateConstInBoundsByteGEP(BaseAddr, Offset);
    }

    RegAddr = BaseAddr.withElementType(MemTy);
  }

  CGF.EmitBranch(ContBlock);

  //=======================================
  // Argument was on the stack
  //=======================================
  CGF.EmitBlock(OnStackBlock);

  Address stack_p = CGF.Builder.CreateStructGEP(VAListAddr, 0, "stack_p");
  llvm::Value *OnStackPtr = CGF.Builder.CreateLoad(stack_p, "stack");

  // Again, stack arguments may need realignment. In this case both integer and
  // floating-point ones might be affected.
  if (!IsIndirect && TyAlign.getQuantity() > 8) {
    OnStackPtr = emitRoundPointerUpToAlignment(CGF, OnStackPtr, TyAlign);
  }
  Address OnStackAddr = Address(OnStackPtr, CGF.Int8Ty,
                                std::max(CharUnits::fromQuantity(8), TyAlign));

  // All stack slots are multiples of 8 bytes.
  CharUnits StackSlotSize = CharUnits::fromQuantity(8);
  CharUnits StackSize;
  if (IsIndirect)
    StackSize = StackSlotSize;
  else
    StackSize = TySize.alignTo(StackSlotSize);

  llvm::Value *StackSizeC = CGF.Builder.getSize(StackSize);
  llvm::Value *NewStack = CGF.Builder.CreateInBoundsGEP(
      CGF.Int8Ty, OnStackPtr, StackSizeC, "new_stack");

  // Write the new value of __stack for the next call to va_arg
  CGF.Builder.CreateStore(NewStack, stack_p);

  if (CGF.CGM.getDataLayout().isBigEndian() && !isAggregateTypeForABI(Ty) &&
      TySize < StackSlotSize) {
    CharUnits Offset = StackSlotSize - TySize;
    OnStackAddr = CGF.Builder.CreateConstInBoundsByteGEP(OnStackAddr, Offset);
  }

  OnStackAddr = OnStackAddr.withElementType(MemTy);

  CGF.EmitBranch(ContBlock);

  //=======================================
  // Tidy up
  //=======================================
  CGF.EmitBlock(ContBlock);

  Address ResAddr = emitMergePHI(CGF, RegAddr, InRegBlock, OnStackAddr,
                                 OnStackBlock, "vaargs.addr");

  if (IsIndirect)
    return CGF.EmitLoadOfAnyValue(
        CGF.MakeAddrLValue(
            Address(CGF.Builder.CreateLoad(ResAddr, "vaarg.addr"), ElementTy,
                    TyAlign),
            Ty),
        Slot);

  return CGF.EmitLoadOfAnyValue(CGF.MakeAddrLValue(ResAddr, Ty), Slot);
}

RValue AArch64ABIInfo::EmitDarwinVAArg(Address VAListAddr, QualType Ty,
                                       CodeGenFunction &CGF,
                                       AggValueSlot Slot) const {
  // The backend's lowering doesn't support va_arg for aggregates or
  // illegal vector types.  Lower VAArg here for these cases and use
  // the LLVM va_arg instruction for everything else.
  if (!isAggregateTypeForABI(Ty) && !isIllegalVectorType(Ty))
    return CGF.EmitLoadOfAnyValue(
        CGF.MakeAddrLValue(
            EmitVAArgInstr(CGF, VAListAddr, Ty, ABIArgInfo::getDirect()), Ty),
        Slot);

  uint64_t PointerSize = getTarget().getPointerWidth(LangAS::Default) / 8;
  CharUnits SlotSize = CharUnits::fromQuantity(PointerSize);

  // Empty records are ignored for parameter passing purposes.
  if (isEmptyRecord(getContext(), Ty, true))
    return Slot.asRValue();

  // The size of the actual thing passed, which might end up just
  // being a pointer for indirect types.
  auto TyInfo = getContext().getTypeInfoInChars(Ty);

  // Arguments bigger than 16 bytes which aren't homogeneous
  // aggregates should be passed indirectly.
  bool IsIndirect = false;
  if (TyInfo.Width.getQuantity() > 16) {
    const Type *Base = nullptr;
    uint64_t Members = 0;
    IsIndirect = !isHomogeneousAggregate(Ty, Base, Members);
  }

  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, IsIndirect, TyInfo, SlotSize,
                          /*AllowHigherAlign*/ true, Slot);
}

RValue AArch64ABIInfo::EmitMSVAArg(CodeGenFunction &CGF, Address VAListAddr,
                                   QualType Ty, AggValueSlot Slot) const {
  bool IsIndirect = false;

  // Composites larger than 16 bytes are passed by reference.
  if (isAggregateTypeForABI(Ty) && getContext().getTypeSize(Ty) > 128)
    IsIndirect = true;

  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, IsIndirect,
                          CGF.getContext().getTypeInfoInChars(Ty),
                          CharUnits::fromQuantity(8),
                          /*allowHigherAlign*/ false, Slot);
}

static bool isStreamingCompatible(const FunctionDecl *F) {
  if (const auto *T = F->getType()->getAs<FunctionProtoType>())
    return T->getAArch64SMEAttributes() &
           FunctionType::SME_PStateSMCompatibleMask;
  return false;
}

// Report an error if an argument or return value of type Ty would need to be
// passed in a floating-point register.
static void diagnoseIfNeedsFPReg(DiagnosticsEngine &Diags,
                                 const StringRef ABIName,
                                 const AArch64ABIInfo &ABIInfo,
                                 const QualType &Ty, const NamedDecl *D,
                                 SourceLocation loc) {
  const Type *HABase = nullptr;
  uint64_t HAMembers = 0;
  if (Ty->isFloatingType() || Ty->isVectorType() ||
      ABIInfo.isHomogeneousAggregate(Ty, HABase, HAMembers)) {
    Diags.Report(loc, diag::err_target_unsupported_type_for_abi)
        << D->getDeclName() << Ty << ABIName;
  }
}

// If we are using a hard-float ABI, but do not have floating point registers,
// then report an error for any function arguments or returns which would be
// passed in floating-pint registers.
void AArch64TargetCodeGenInfo::checkFunctionABI(
    CodeGenModule &CGM, const FunctionDecl *FuncDecl) const {
  const AArch64ABIInfo &ABIInfo = getABIInfo<AArch64ABIInfo>();
  const TargetInfo &TI = ABIInfo.getContext().getTargetInfo();

  if (!TI.hasFeature("fp") && !ABIInfo.isSoftFloat()) {
    diagnoseIfNeedsFPReg(CGM.getDiags(), TI.getABI(), ABIInfo,
                         FuncDecl->getReturnType(), FuncDecl,
                         FuncDecl->getLocation());
    for (ParmVarDecl *PVD : FuncDecl->parameters()) {
      diagnoseIfNeedsFPReg(CGM.getDiags(), TI.getABI(), ABIInfo, PVD->getType(),
                           PVD, FuncDecl->getLocation());
    }
  }
}

enum class ArmSMEInlinability : uint8_t {
  Ok = 0,
  ErrorCalleeRequiresNewZA = 1 << 0,
  ErrorCalleeRequiresNewZT0 = 1 << 1,
  WarnIncompatibleStreamingModes = 1 << 2,
  ErrorIncompatibleStreamingModes = 1 << 3,

  IncompatibleStreamingModes =
      WarnIncompatibleStreamingModes | ErrorIncompatibleStreamingModes,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/ErrorIncompatibleStreamingModes),
};

/// Determines if there are any Arm SME ABI issues with inlining \p Callee into
/// \p Caller. Returns the issue (if any) in the ArmSMEInlinability bit enum.
static ArmSMEInlinability GetArmSMEInlinability(const FunctionDecl *Caller,
                                                const FunctionDecl *Callee) {
  bool CallerIsStreaming =
      IsArmStreamingFunction(Caller, /*IncludeLocallyStreaming=*/true);
  bool CalleeIsStreaming =
      IsArmStreamingFunction(Callee, /*IncludeLocallyStreaming=*/true);
  bool CallerIsStreamingCompatible = isStreamingCompatible(Caller);
  bool CalleeIsStreamingCompatible = isStreamingCompatible(Callee);

  ArmSMEInlinability Inlinability = ArmSMEInlinability::Ok;

  if (!CalleeIsStreamingCompatible &&
      (CallerIsStreaming != CalleeIsStreaming || CallerIsStreamingCompatible)) {
    if (CalleeIsStreaming)
      Inlinability |= ArmSMEInlinability::ErrorIncompatibleStreamingModes;
    else
      Inlinability |= ArmSMEInlinability::WarnIncompatibleStreamingModes;
  }
  if (auto *NewAttr = Callee->getAttr<ArmNewAttr>()) {
    if (NewAttr->isNewZA())
      Inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZA;
    if (NewAttr->isNewZT0())
      Inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZT0;
  }

  return Inlinability;
}

void AArch64TargetCodeGenInfo::checkFunctionCallABIStreaming(
    CodeGenModule &CGM, SourceLocation CallLoc, const FunctionDecl *Caller,
    const FunctionDecl *Callee) const {
  if (!Caller || !Callee || !Callee->hasAttr<AlwaysInlineAttr>())
    return;

  ArmSMEInlinability Inlinability = GetArmSMEInlinability(Caller, Callee);

  if ((Inlinability & ArmSMEInlinability::IncompatibleStreamingModes) !=
      ArmSMEInlinability::Ok)
    CGM.getDiags().Report(
        CallLoc,
        (Inlinability & ArmSMEInlinability::ErrorIncompatibleStreamingModes) ==
                ArmSMEInlinability::ErrorIncompatibleStreamingModes
            ? diag::err_function_always_inline_attribute_mismatch
            : diag::warn_function_always_inline_attribute_mismatch)
        << Caller->getDeclName() << Callee->getDeclName() << "streaming";

  if ((Inlinability & ArmSMEInlinability::ErrorCalleeRequiresNewZA) ==
      ArmSMEInlinability::ErrorCalleeRequiresNewZA)
    CGM.getDiags().Report(CallLoc, diag::err_function_always_inline_new_za)
        << Callee->getDeclName();

  if ((Inlinability & ArmSMEInlinability::ErrorCalleeRequiresNewZT0) ==
      ArmSMEInlinability::ErrorCalleeRequiresNewZT0)
    CGM.getDiags().Report(CallLoc, diag::err_function_always_inline_new_zt0)
        << Callee->getDeclName();
}

// If the target does not have floating-point registers, but we are using a
// hard-float ABI, there is no way to pass floating-point, vector or HFA values
// to functions, so we report an error.
void AArch64TargetCodeGenInfo::checkFunctionCallABISoftFloat(
    CodeGenModule &CGM, SourceLocation CallLoc, const FunctionDecl *Caller,
    const FunctionDecl *Callee, const CallArgList &Args,
    QualType ReturnType) const {
  const AArch64ABIInfo &ABIInfo = getABIInfo<AArch64ABIInfo>();
  const TargetInfo &TI = ABIInfo.getContext().getTargetInfo();

  if (!Caller || TI.hasFeature("fp") || ABIInfo.isSoftFloat())
    return;

  diagnoseIfNeedsFPReg(CGM.getDiags(), TI.getABI(), ABIInfo, ReturnType,
                       Callee ? Callee : Caller, CallLoc);

  for (const CallArg &Arg : Args)
    diagnoseIfNeedsFPReg(CGM.getDiags(), TI.getABI(), ABIInfo, Arg.getType(),
                         Callee ? Callee : Caller, CallLoc);
}

void AArch64TargetCodeGenInfo::checkFunctionCallABI(CodeGenModule &CGM,
                                                    SourceLocation CallLoc,
                                                    const FunctionDecl *Caller,
                                                    const FunctionDecl *Callee,
                                                    const CallArgList &Args,
                                                    QualType ReturnType) const {
  checkFunctionCallABIStreaming(CGM, CallLoc, Caller, Callee);
  checkFunctionCallABISoftFloat(CGM, CallLoc, Caller, Callee, Args, ReturnType);
}

bool AArch64TargetCodeGenInfo::wouldInliningViolateFunctionCallABI(
    const FunctionDecl *Caller, const FunctionDecl *Callee) const {
  return Caller && Callee &&
         GetArmSMEInlinability(Caller, Callee) != ArmSMEInlinability::Ok;
}

void AArch64ABIInfo::appendAttributeMangling(TargetClonesAttr *Attr,
                                             unsigned Index,
                                             raw_ostream &Out) const {
  appendAttributeMangling(Attr->getFeatureStr(Index), Out);
}

void AArch64ABIInfo::appendAttributeMangling(StringRef AttrStr,
                                             raw_ostream &Out) const {
  if (AttrStr == "default") {
    Out << ".default";
    return;
  }

  Out << "._";
  SmallVector<StringRef, 8> Features;
  AttrStr.split(Features, "+");
  for (auto &Feat : Features)
    Feat = Feat.trim();

  llvm::sort(Features, [](const StringRef LHS, const StringRef RHS) {
    return LHS.compare(RHS) < 0;
  });

  llvm::SmallDenseSet<StringRef, 8> UniqueFeats;
  for (auto &Feat : Features)
    if (auto Ext = llvm::AArch64::parseFMVExtension(Feat))
      if (UniqueFeats.insert(Ext->Name).second)
        Out << 'M' << Ext->Name;
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createAArch64TargetCodeGenInfo(CodeGenModule &CGM,
                                        AArch64ABIKind Kind) {
  return std::make_unique<AArch64TargetCodeGenInfo>(CGM.getTypes(), Kind);
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createWindowsAArch64TargetCodeGenInfo(CodeGenModule &CGM,
                                               AArch64ABIKind K) {
  return std::make_unique<WindowsAArch64TargetCodeGenInfo>(CGM.getTypes(), K);
}
