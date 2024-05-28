//====- LoweringPrepareArm64CXXABI.cpp - Arm64 ABI specific code -----====//
//
// Part of the LLVM Project,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
//
// This file provides ARM64 C++ ABI specific code that is used during LLVMIR
// lowering prepare.
//
//===------------------------------------------------------------------===//

#include "../IR/MissingFeatures.h"
#include "LoweringPrepareItaniumCXXABI.h"
#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include <assert.h>

using cir::AArch64ABIKind;

namespace {
class LoweringPrepareAArch64CXXABI : public LoweringPrepareItaniumCXXABI {
public:
  LoweringPrepareAArch64CXXABI(AArch64ABIKind k) : Kind(k) {}
  mlir::Value lowerVAArg(cir::CIRBaseBuilderTy &builder, mlir::cir::VAArgOp op,
                         const cir::CIRDataLayout &datalayout) override;

private:
  AArch64ABIKind Kind;
  mlir::Value lowerAAPCSVAArg(cir::CIRBaseBuilderTy &builder,
                              mlir::cir::VAArgOp op,
                              const cir::CIRDataLayout &datalayout);
  bool isDarwinPCS() const { return Kind == AArch64ABIKind::DarwinPCS; }
  mlir::Value lowerMSVAArg(cir::CIRBaseBuilderTy &builder,
                           mlir::cir::VAArgOp op,
                           const cir::CIRDataLayout &datalayout) {
    llvm_unreachable("MSVC ABI not supported yet");
  }
  mlir::Value lowerDarwinVAArg(cir::CIRBaseBuilderTy &builder,
                               mlir::cir::VAArgOp op,
                               const cir::CIRDataLayout &datalayout) {
    llvm_unreachable("Darwin ABI not supported yet");
  }
};
} // namespace

cir::LoweringPrepareCXXABI *
cir::LoweringPrepareCXXABI::createAArch64ABI(AArch64ABIKind k) {
  return new LoweringPrepareAArch64CXXABI(k);
}

mlir::Value LoweringPrepareAArch64CXXABI::lowerAAPCSVAArg(
    cir::CIRBaseBuilderTy &builder, mlir::cir::VAArgOp op,
    const cir::CIRDataLayout &datalayout) {
  auto loc = op->getLoc();
  auto valist = op->getOperand(0);
  auto opResTy = op.getType();
  // front end should not produce non-scalar type of VAArgOp
  bool isSupportedType =
      opResTy.isa<mlir::cir::IntType, mlir::cir::SingleType,
                  mlir::cir::PointerType, mlir::cir::BoolType,
                  mlir::cir::DoubleType, mlir::cir::ArrayType>();

  // Homogenous Aggregate type not supported and indirect arg
  // passing not supported yet. And for these supported types,
  // we should not have alignment greater than 8 problem.
  assert(isSupportedType);
  assert(!cir::MissingFeatures::classifyArgumentTypeForAArch64());
  // indirect arg passing would expect one more level of pointer dereference.
  assert(!cir::MissingFeatures::handleAArch64Indirect());
  // false as a place holder for now, as we don't have a way to query
  bool isIndirect = false;
  assert(!cir::MissingFeatures::supportgetCoerceToTypeForAArch64());
  // we don't convert to LLVM Type here as we are lowering to CIR here.
  // so baseTy is the just type of the result of va_arg.
  // but it depends on arg type indirectness and coercion defined by ABI.
  auto baseTy = opResTy;

  if (baseTy.isa<mlir::cir::ArrayType>()) {
    llvm_unreachable("ArrayType VAArg loweing NYI");
  }
  // numRegs may not be 1 if ArrayType is supported.
  unsigned numRegs = 1;

  if (Kind == AArch64ABIKind::AAPCSSoft) {
    llvm_unreachable("AAPCSSoft cir.var_arg lowering NYI");
  }
  bool IsFPR = mlir::cir::isAnyFloatingPointType(baseTy);

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
  auto curInsertionP = builder.saveInsertionPoint();
  auto currentBlock = builder.getInsertionBlock();
  auto boolTy = builder.getBoolTy();

  auto maybeRegBlock = builder.createBlock(builder.getBlock()->getParent());
  auto inRegBlock = builder.createBlock(builder.getBlock()->getParent());
  auto onStackBlock = builder.createBlock(builder.getBlock()->getParent());

  //=======================================
  // Find out where argument was passed
  //=======================================

  // If v/gr_offs >= 0 we're already using the stack for this type of
  // argument. We don't want to keep updating regOffs (in case it overflows,
  // though anyone passing 2GB of arguments, each at most 16 bytes, deserves
  // whatever they get).

  assert(!cir::MissingFeatures::supportTySizeQueryForAArch64());
  assert(!cir::MissingFeatures::supportTyAlignQueryForAArch64());
  // One is just place holder for now, as we don't have a way to query
  // type size and alignment.
  clang::CharUnits tySize =
      clang::CharUnits::fromQuantity(datalayout.getTypeStoreSize(opResTy));
  clang::CharUnits tyAlign =
      clang::CharUnits::fromQuantity(datalayout.getAlignment(opResTy, true));

  // indirectness, type size and type alignment all
  // decide regSize, but they are all ABI defined
  // thus need ABI lowering query system.
  assert(!cir::MissingFeatures::handleAArch64Indirect());
  int regSize = isIndirect ? 8 : tySize.getQuantity();
  int regTopIndex;
  mlir::Value regOffsP;
  mlir::cir::LoadOp regOffs;

  builder.restoreInsertionPoint(curInsertionP);
  // 3 is the field number of __gr_offs, 4 is the field number of __vr_offs
  if (!IsFPR) {
    regOffsP = builder.createGetMemberOp(loc, valist, "gr_offs", 3);
    regOffs = builder.create<mlir::cir::LoadOp>(loc, regOffsP);
    regTopIndex = 1;
    regSize = llvm::alignTo(regSize, 8);
  } else {
    regOffsP = builder.createGetMemberOp(loc, valist, "vr_offs", 4);
    regOffs = builder.create<mlir::cir::LoadOp>(loc, regOffsP);
    regTopIndex = 2;
    regSize = 16 * numRegs;
  }

  //=======================================
  // Find out where argument was passed
  //=======================================

  // If regOffs >= 0 we're already using the stack for this type of
  // argument. We don't want to keep updating regOffs (in case it overflows,
  // though anyone passing 2GB of arguments, each at most 16 bytes, deserves
  // whatever they get).
  auto zeroValue = builder.create<mlir::cir::ConstantOp>(
      loc, regOffs.getType(), mlir::cir::IntAttr::get(regOffs.getType(), 0));
  auto usingStack = builder.create<mlir::cir::CmpOp>(
      loc, boolTy, mlir::cir::CmpOpKind::ge, regOffs, zeroValue);
  builder.create<mlir::cir::BrCondOp>(loc, usingStack, onStackBlock,
                                      maybeRegBlock);

  auto contBlock = currentBlock->splitBlock(op);

  // Otherwise, at least some kind of argument could go in these registers, the
  // question is whether this particular type is too big.
  builder.setInsertionPointToEnd(maybeRegBlock);

  // Integer arguments may need to correct register alignment (for example a
  // "struct { __int128 a; };" gets passed in x_2N, x_{2N+1}). In this case we
  // align __gr_offs to calculate the potential address.
  if (!IsFPR && !isIndirect && tyAlign.getQuantity() > 8) {
    assert(!cir::MissingFeatures::handleAArch64Indirect());
    assert(!cir::MissingFeatures::supportTyAlignQueryForAArch64());
    llvm_unreachable("register alignment correction NYI");
  }

  // Update the gr_offs/vr_offs pointer for next call to va_arg on this va_list.
  // The fact that this is done unconditionally reflects the fact that
  // allocating an argument to the stack also uses up all the remaining
  // registers of the appropriate kind.
  auto regSizeValue = builder.create<mlir::cir::ConstantOp>(
      loc, regOffs.getType(),
      mlir::cir::IntAttr::get(regOffs.getType(), regSize));
  auto newOffset = builder.create<mlir::cir::BinOp>(
      loc, regOffs.getType(), mlir::cir::BinOpKind::Add, regOffs, regSizeValue);
  builder.createStore(loc, newOffset, regOffsP);
  // Now we're in a position to decide whether this argument really was in
  // registers or not.
  auto inRegs = builder.create<mlir::cir::CmpOp>(
      loc, boolTy, mlir::cir::CmpOpKind::le, newOffset, zeroValue);
  builder.create<mlir::cir::BrCondOp>(loc, inRegs, inRegBlock, onStackBlock);

  //=======================================
  // Argument was in registers
  //=======================================
  // Now we emit the code for if the argument was originally passed in
  // registers. First start the appropriate block:
  builder.setInsertionPointToEnd(inRegBlock);
  auto regTopP = builder.createGetMemberOp(
      loc, valist, IsFPR ? "vr_top" : "gr_top", regTopIndex);
  auto regTop = builder.create<mlir::cir::LoadOp>(loc, regTopP);
  auto i8Ty = mlir::IntegerType::get(builder.getContext(), 8);
  auto i8PtrTy = mlir::cir::PointerType::get(builder.getContext(), i8Ty);
  auto castRegTop = builder.createBitcast(regTop, i8PtrTy);
  auto resAsInt8P = builder.create<mlir::cir::PtrStrideOp>(
      loc, castRegTop.getType(), castRegTop, regOffs);

  if (isIndirect) {
    assert(!cir::MissingFeatures::handleAArch64Indirect());
    llvm_unreachable("indirect arg passing NYI");
  }

  // TODO: isHFA, numMembers and base should be query result from query
  uint64_t numMembers = 0;
  assert(!cir::MissingFeatures::supportisHomogeneousAggregateQueryForAArch64());
  bool isHFA = false;
  // though endianess can be known from datalayout, it might need an unified
  // ABI lowering query system to answer the question.
  assert(!cir::MissingFeatures::supportisEndianQueryForAArch64());
  bool isBigEndian = datalayout.isBigEndian();
  assert(!cir::MissingFeatures::supportisAggregateTypeForABIAArch64());
  // TODO: isAggregateTypeForABI should be query result from ABI info
  bool isAggregateTypeForABI = false;
  if (isHFA && numMembers > 1) {
    // Homogeneous aggregates passed in registers will have their elements split
    // and stored 16-bytes apart regardless of size (they're notionally in qN,
    // qN+1, ...). We reload and store into a temporary local variable
    // contiguously.
    assert(!isIndirect && "Homogeneous aggregates should be passed directly");
    llvm_unreachable("Homogeneous aggregates NYI");
  } else {
    assert(!cir::MissingFeatures::supportTyAlignQueryForAArch64());
    // TODO: slotSize should be query result about alignment.
    clang::CharUnits slotSize = clang::CharUnits::fromQuantity(8);
    if (isBigEndian && !isIndirect && (isHFA || isAggregateTypeForABI) &&
        tySize < slotSize) {
      clang::CharUnits offset = slotSize - tySize;
      auto offsetConst = builder.create<mlir::cir::ConstantOp>(
          loc, regOffs.getType(),
          mlir::cir::IntAttr::get(regOffs.getType(), offset.getQuantity()));

      resAsInt8P = builder.create<mlir::cir::PtrStrideOp>(
          loc, castRegTop.getType(), resAsInt8P, offsetConst);
    }
  }

  auto resAsVoidP = builder.createBitcast(resAsInt8P, regTop.getType());

  // On big-endian platforms, the value will be right-aligned in its stack slot.
  // and we also need to think about other ABI lowering concerns listed below.
  assert(!cir::MissingFeatures::handleBigEndian());
  assert(!cir::MissingFeatures::handleAArch64Indirect());
  assert(!cir::MissingFeatures::supportisHomogeneousAggregateQueryForAArch64());
  assert(!cir::MissingFeatures::supportTySizeQueryForAArch64());
  assert(!cir::MissingFeatures::supportTyAlignQueryForAArch64());

  builder.create<mlir::cir::BrOp>(loc, mlir::ValueRange{resAsVoidP}, contBlock);

  //=======================================
  // Argument was on the stack
  //=======================================
  builder.setInsertionPointToEnd(onStackBlock);
  auto stackP = builder.createGetMemberOp(loc, valist, "stack", 0);

  auto onStackPtr = builder.create<mlir::cir::LoadOp>(loc, stackP);
  auto ptrDiffTy =
      mlir::cir::IntType::get(builder.getContext(), 64, /*signed=*/false);

  assert(!cir::MissingFeatures::handleAArch64Indirect());
  assert(!cir::MissingFeatures::supportTyAlignQueryForAArch64());
  // Again, stack arguments may need realignment. In this case both integer and
  // floating-point ones might be affected.
  if (!isIndirect && tyAlign.getQuantity() > 8) {
    // TODO: this algorithm requres casting from ptr type to int type, then
    // back to ptr type thus needs careful handling. NYI now.
    llvm_unreachable("alignment greater than 8 NYI");
  }

  // All stack slots are multiples of 8 bytes.
  clang::CharUnits stackSlotSize = clang::CharUnits::fromQuantity(8);
  clang::CharUnits stackSize;
  if (isIndirect)
    stackSize = stackSlotSize;
  else
    stackSize = tySize.alignTo(stackSlotSize);

  // On big-endian platforms, the value will be right-aligned in its stack slot
  // Also, the consideration involves type size and alignment, arg indirectness
  // which are all ABI defined thus need ABI lowering query system.
  // The implementation we have now supports most common cases which assumes
  // no indirectness, no alignment greater than 8, and little endian.
  assert(!cir::MissingFeatures::handleBigEndian());
  assert(!cir::MissingFeatures::supportTySizeQueryForAArch64());

  auto stackSizeC = builder.create<mlir::cir::ConstantOp>(
      loc, ptrDiffTy,
      mlir::cir::IntAttr::get(ptrDiffTy, stackSize.getQuantity()));
  auto castStack = builder.createBitcast(onStackPtr, i8PtrTy);
  // Write the new value of __stack for the next call to va_arg
  auto newStackAsi8Ptr = builder.create<mlir::cir::PtrStrideOp>(
      loc, castStack.getType(), castStack, stackSizeC);
  auto newStack = builder.createBitcast(newStackAsi8Ptr, onStackPtr.getType());
  builder.createStore(loc, newStack, stackP);

  if (isBigEndian && !isAggregateTypeForABI && tySize < stackSlotSize) {
    clang::CharUnits offset = stackSlotSize - tySize;
    auto offsetConst = builder.create<mlir::cir::ConstantOp>(
        loc, ptrDiffTy,
        mlir::cir::IntAttr::get(ptrDiffTy, offset.getQuantity()));
    auto offsetStackAsi8Ptr = builder.create<mlir::cir::PtrStrideOp>(
        loc, castStack.getType(), castStack, offsetConst);
    auto onStackPtrBE =
        builder.createBitcast(offsetStackAsi8Ptr, onStackPtr.getType());
    builder.create<mlir::cir::BrOp>(loc, mlir::ValueRange{onStackPtrBE},
                                    contBlock);
  } else {
    builder.create<mlir::cir::BrOp>(loc, mlir::ValueRange{onStackPtr},
                                    contBlock);
  }

  // generate additional instructions for end block
  builder.setInsertionPoint(op);
  contBlock->addArgument(onStackPtr.getType(), loc);
  auto resP = contBlock->getArgument(0);
  assert(resP.getType().isa<mlir::cir::PointerType>());
  auto opResPTy = mlir::cir::PointerType::get(builder.getContext(), opResTy);
  auto castResP = builder.createBitcast(resP, opResPTy);
  auto res = builder.create<mlir::cir::LoadOp>(loc, castResP);
  // there would be another level of ptr dereference if indirect arg passing
  assert(!cir::MissingFeatures::handleAArch64Indirect());
  if (isIndirect) {
    res = builder.create<mlir::cir::LoadOp>(loc, res.getResult());
  }
  return res.getResult();
}

mlir::Value
LoweringPrepareAArch64CXXABI::lowerVAArg(cir::CIRBaseBuilderTy &builder,
                                         mlir::cir::VAArgOp op,
                                         const cir::CIRDataLayout &datalayout) {
  return Kind == AArch64ABIKind::Win64 ? lowerMSVAArg(builder, op, datalayout)
         : isDarwinPCS() ? lowerDarwinVAArg(builder, op, datalayout)
                         : lowerAAPCSVAArg(builder, op, datalayout);
}
