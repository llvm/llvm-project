//===- Xtensa.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;
//===----------------------------------------------------------------------===//
// Xtensa ABI Implementation
//===----------------------------------------------------------------------===//

namespace {
class XtensaABIInfo : public DefaultABIInfo {
private:
  static const int MaxNumArgGPRs = 6;
  static const int MaxNumRetGPRs = 4;

public:
  XtensaABIInfo(CodeGen::CodeGenTypes &CGT) : DefaultABIInfo(CGT) {}

  // DefaultABIInfo's classifyReturnType and classifyArgumentType are
  // non-virtual, but computeInfo is virtual, so we overload it.
  void computeInfo(CGFunctionInfo &FI) const override;

  ABIArgInfo classifyArgumentType(QualType Ty, int &ArgGPRsLeft) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;

  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;

  ABIArgInfo extendType(QualType Ty) const;
};
} // end anonymous namespace

void XtensaABIInfo::computeInfo(CGFunctionInfo &FI) const {
  QualType RetTy = FI.getReturnType();
  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(RetTy);

  int ArgGPRsLeft = MaxNumArgGPRs;
  for (auto &ArgInfo : FI.arguments()) {
    ArgInfo.info = classifyArgumentType(ArgInfo.type, ArgGPRsLeft);
  }
}

ABIArgInfo XtensaABIInfo::classifyArgumentType(QualType Ty,
                                               int &ArgGPRsLeft) const {
  assert(ArgGPRsLeft <= MaxNumArgGPRs && "Arg GPR tracking underflow");
  Ty = useFirstFieldIfTransparentUnion(Ty);
  // Structures with either a non-trivial destructor or a non-trivial
  // copy constructor are always passed indirectly.
  if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI())) {
    if (ArgGPRsLeft)
      ArgGPRsLeft -= 1;
    return getNaturalAlignIndirect(Ty, getDataLayout().getAllocaAddrSpace(),
                                   /*ByVal=*/RAA ==
                                       CGCXXABI::RAA_DirectInMemory);
  }

  // Ignore empty structs/unions.
  if (isEmptyRecord(getContext(), Ty, true))
    return ABIArgInfo::getIgnore();

  uint64_t Size = getContext().getTypeSize(Ty);
  uint64_t NeededAlign = getContext().getTypeAlign(Ty);
  bool MustUseStack = false;
  int NeededArgGPRs = (Size + 31) / 32;

  if (NeededAlign == (2 * 32))
    NeededArgGPRs += (ArgGPRsLeft % 2);

  // Put on stack objects which are not fit to 6 registers,
  // also on stack object which alignment more then 16 bytes and
  // object with 16-byte alignment if it isn't the first argument.
  if ((NeededArgGPRs > ArgGPRsLeft) || (NeededAlign > (4 * 32)) ||
      ((ArgGPRsLeft < 6) && (NeededAlign == (4 * 32)))) {
    MustUseStack = true;
    NeededArgGPRs = ArgGPRsLeft;
  }
  ArgGPRsLeft -= NeededArgGPRs;

  if (!isAggregateTypeForABI(Ty) && !Ty->isVectorType() && !MustUseStack) {
    // Treat an enum type as its underlying type.
    if (const auto *ED = Ty->getAsEnumDecl())
      Ty = ED->getIntegerType();
    // All integral types are promoted to XLen width, unless passed on the
    // stack.
    if (Size < 32 && Ty->isIntegralOrEnumerationType() && !MustUseStack) {
      return extendType(Ty);
    }
    // Assume that type has 32, 64 or 128 bits
    return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), Size));
  }

  // Aggregates which are <= 6*32 will be passed in registers if possible,
  // so coerce to integers.
  if ((Size <= (MaxNumArgGPRs * 32)) && (!MustUseStack)) {
    if (Size <= 32) {
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), 32));
    } else if (NeededAlign == (2 * 32)) {
      return ABIArgInfo::getDirect(llvm::ArrayType::get(
          llvm::IntegerType::get(getVMContext(), 64), NeededArgGPRs / 2));
    } else if (NeededAlign == (4 * 32)) {
      return ABIArgInfo::getDirect(llvm::IntegerType::get(getVMContext(), 128));
    } else {
      return ABIArgInfo::getDirect(llvm::ArrayType::get(
          llvm::IntegerType::get(getVMContext(), 32), NeededArgGPRs));
    }
  }
#undef MAX_STRUCT_IN_REGS_SIZE
  return getNaturalAlignIndirect(Ty, getDataLayout().getAllocaAddrSpace(),
                                 /*ByVal=*/true);
}

ABIArgInfo XtensaABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  int ArgGPRsLeft = MaxNumRetGPRs;
  auto RetSize = llvm::alignTo(getContext().getTypeSize(RetTy), 32) / 32;

  // The rules for return and argument with type size more then 4 bytes
  // are the same, so defer to classifyArgumentType.
  if (RetSize > 1)
    return classifyArgumentType(RetTy, ArgGPRsLeft);

  return DefaultABIInfo::classifyReturnType(RetTy);
}

RValue XtensaABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                                QualType Ty, AggValueSlot Slot) const {
  // The va_list structure memory layout:
  // struct __va_list_tag {
  //   int32_t *va_stk;
  //   int32_t *va_reg;
  //   int32_t va_ndx;
  // };
  CGBuilderTy &Builder = CGF.Builder;

  Address OverflowAreaPtr = Builder.CreateStructGEP(VAListAddr, 0, "__va_stk");
  Address OverflowArea = Address(Builder.CreateLoad(OverflowAreaPtr, ""),
                                 CGF.Int32Ty, CharUnits::fromQuantity(4));
  Address RegSaveAreaPtr = Builder.CreateStructGEP(VAListAddr, 1, "__va_reg");
  Address RegSaveArea = Address(Builder.CreateLoad(RegSaveAreaPtr, ""),
                                CGF.Int32Ty, CharUnits::fromQuantity(4));
  Address ARAreaPtr = Builder.CreateStructGEP(VAListAddr, 2, "__va_ndx");
  llvm::Value *ARIndex = Builder.CreateLoad(ARAreaPtr, "");

  ARIndex = Builder.CreateLShr(ARIndex, Builder.getInt32(2));

  unsigned Align = getContext().getTypeAlign(Ty) / 32;
  unsigned Size = (getContext().getTypeSize(Ty) + 31) / 32;

  if (Align > 1) {
    ARIndex = Builder.CreateAdd(ARIndex, Builder.getInt32(Align - 1));
    ARIndex =
        Builder.CreateAnd(ARIndex, Builder.getInt32((uint32_t) ~(Align - 1)));
  }

  llvm::Value *ARIndexNext = Builder.CreateAdd(ARIndex, Builder.getInt32(Size));
  Builder.CreateStore(Builder.CreateShl(ARIndexNext, Builder.getInt32(2)),
                      ARAreaPtr);

  const unsigned OverflowLimit = 6;
  llvm::Value *CC = Builder.CreateICmpULE(
      ARIndexNext, Builder.getInt32(OverflowLimit), "cond");

  llvm::BasicBlock *UsingRegSaveArea =
      CGF.createBasicBlock("using_regsavearea");
  llvm::BasicBlock *UsingOverflow = CGF.createBasicBlock("using_overflow");
  llvm::BasicBlock *Cont = CGF.createBasicBlock("cont");

  Builder.CreateCondBr(CC, UsingRegSaveArea, UsingOverflow);

  llvm::Type *DirectTy = CGF.ConvertType(Ty);

  // Case 1: consume registers.
  Address RegAddr = Address::invalid();
  {
    CGF.EmitBlock(UsingRegSaveArea);

    CharUnits RegSize = CharUnits::fromQuantity(4);
    RegSaveArea =
        Address(Builder.CreateInBoundsGEP(
                    CGF.Int32Ty, RegSaveArea.emitRawPointer(CGF), ARIndex),
                CGF.Int32Ty,
                RegSaveArea.getAlignment().alignmentOfArrayElement(RegSize));
    RegAddr = RegSaveArea.withElementType(DirectTy);
    CGF.EmitBranch(Cont);
  }

  // Case 2: consume space in the overflow area.
  Address MemAddr = Address::invalid();
  {
    CGF.EmitBlock(UsingOverflow);
    llvm::Value *CC1 = Builder.CreateICmpULE(
        ARIndex, Builder.getInt32(OverflowLimit), "cond_overflow");

    llvm::Value *ARIndexOff = Builder.CreateSelect(
        CC1, Builder.CreateSub(Builder.getInt32(8), ARIndex),
        Builder.getInt32(0));

    llvm::Value *ARIndexCorr = Builder.CreateAdd(ARIndex, ARIndexOff);
    llvm::Value *ARIndexNextCorr = Builder.CreateAdd(ARIndexNext, ARIndexOff);
    Builder.CreateStore(Builder.CreateShl(ARIndexNextCorr, Builder.getInt32(2)),
                        ARAreaPtr);

    CharUnits RegSize = CharUnits::fromQuantity(4);
    OverflowArea =
        Address(Builder.CreateInBoundsGEP(
                    CGF.Int32Ty, OverflowArea.emitRawPointer(CGF), ARIndexCorr),
                CGF.Int32Ty,
                OverflowArea.getAlignment().alignmentOfArrayElement(RegSize));
    MemAddr = OverflowArea.withElementType(DirectTy);
    CGF.EmitBranch(Cont);
  }

  CGF.EmitBlock(Cont);

  // Merge the cases with a phi.
  Address Result =
      emitMergePHI(CGF, RegAddr, UsingRegSaveArea, MemAddr, UsingOverflow, "");

  return CGF.EmitLoadOfAnyValue(CGF.MakeAddrLValue(Result, Ty), Slot);
}

ABIArgInfo XtensaABIInfo::extendType(QualType Ty) const {
  return ABIArgInfo::getExtend(Ty);
}

namespace {
class XtensaTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  XtensaTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<XtensaABIInfo>(CGT)) {}
};
} // namespace

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createXtensaTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<XtensaTargetCodeGenInfo>(CGM.getTypes());
}
